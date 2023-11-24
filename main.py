import sys
import datetime
import random
import numpy as np
import time
import torch
import torch.backends.cudnn as cudnn
import json
import optuna
import pprint

from pathlib import Path
from torch.utils.tensorboard import SummaryWriter

from timm.data import Mixup
from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy
from timm.scheduler import create_scheduler
from timm.optim import create_optimizer
from timm.utils import NativeScaler, get_state_dict, ModelEma

from engine import train_one_epoch, evaluate
import utils.deit_util as utils
from datasets import get_loaders
from utils.args import get_args_parser
from models import get_model, Meta_mini

from copy import deepcopy
from collections import deque


def main(args):
    utils.init_distributed_mode(args)
    if args.pretrain:
        args.dataset = "meta_dataset"
        args.base_sources = ["ilsvrc_2012"]
        args.val_sources = ["ilsvrc_2012"]
        args.test_sources = ["ilsvrc_2012"]
        args.contrastive_lr = 0.0
        args.contrastive_steps = 0
    print(args)
    if args.contrastive_lr == 0.0 or args.contrastive_steps == 0:
        print("No contrastive learning")
    device = torch.device(args.device)
    args.device = device
    # fix the seed for reproducibility
    seed = args.seed + utils.get_rank()
    args.seed = seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    cudnn.benchmark = True

    args.transform_type = (
        None
        if args.contrastive_lr == 0 or args.contrastive_steps == 0
        else "no_transform"
    )

    if not args.optuna:
        output_dir = Path(args.output_dir)
        if utils.is_main_process():
            try:
                output_dir.mkdir(parents=True, exist_ok=False)
            except FileExistsError:
                input(
                    "Warning: Output directory already exists and is not empty. Press Enter to continue..."
                )
                output_dir.mkdir(parents=True, exist_ok=True)
            with (output_dir / "log.txt").open("a") as f:
                f.write(" ".join(sys.argv) + "\n")
        ##############################################
        # Data loaders
        num_tasks = utils.get_world_size()
        global_rank = utils.get_rank()
        data_loader_train, data_loader_val = get_loaders(
            args, num_tasks, global_rank, transform_type=args.transform_type
        )
        # data_loader_contrastive_train, data_loader_contrastive_val = get_loaders(args, num_tasks, global_rank)
        ##############################################
        # Mixup regularization (by default OFF)
        mixup_fn = None
        mixup_active = (
            args.mixup > 0 or args.cutmix > 0.0 or args.cutmix_minmax is not None
        )
        if mixup_active:
            mixup_fn = Mixup(
                mixup_alpha=args.mixup,
                cutmix_alpha=args.cutmix,
                cutmix_minmax=args.cutmix_minmax,
                prob=args.mixup_prob,
                switch_prob=args.mixup_switch_prob,
                mode=args.mixup_mode,
                label_smoothing=args.smoothing,
                num_classes=args.nClsEpisode,
            )

        ##############################################
        # Model
        print(f"Creating model: ProtoNet {args.arch}")

        model = get_model(args)
        model = model.to(device)

        model_ema = None  # (by default OFF)
        if args.model_ema:
            # Important to create EMA model after cuda(), DP wrapper, and AMP but before SyncBN and DDP wrapper
            model_ema = ModelEma(
                model,
                decay=args.model_ema_decay,
                device="cpu" if args.model_ema_force_cpu else "",
                resume="",
            )

        model_without_ddp = model
        if args.distributed:
            model = torch.nn.parallel.DistributedDataParallel(
                model, device_ids=[args.gpu], find_unused_parameters=args.unused_params
            )
            model_without_ddp = model.module
        n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print("number of params:", n_parameters)

        ##############################################
        # Optimizer & scheduler & criterion
        if args.fp16:
            scale = 1 / 8  # the default lr is for 8 GPUs
            linear_scaled_lr = args.lr * utils.get_world_size() * scale
            args.lr = linear_scaled_lr

        loss_scaler = NativeScaler() if args.fp16 else None

        # optimizer = create_optimizer(args, model_without_ddp)
        if isinstance(model, Meta_mini):
            optimizer = model.meta_optim
            lr_scheduler = model.scheduler
        else:
            optimizer = torch.optim.SGD(
                [p for p in model_without_ddp.parameters() if p.requires_grad],
                args.lr,
                momentum=args.momentum,
                weight_decay=args.weight_decay,  # no weight decay for fine-tuning
            )

            lr_scheduler, _ = create_scheduler(args, optimizer)

        if args.mixup > 0.0:
            # smoothing is handled with mixup label transform
            criterion = SoftTargetCrossEntropy()
        elif args.smoothing:
            criterion = LabelSmoothingCrossEntropy(smoothing=args.smoothing)
        else:
            criterion = torch.nn.CrossEntropyLoss()

        ##############################################
        # Resume training from ckpt (model, optimizer, lr_scheduler, epoch, model_ema, scaler)
        if args.resume:
            if args.resume.startswith("https"):
                checkpoint = torch.hub.load_state_dict_from_url(
                    args.resume, map_location="cpu", check_hash=True
                )
            else:
                checkpoint = torch.load(args.resume, map_location="cpu")

            model_without_ddp.load_state_dict(checkpoint["model"])

            if (
                not args.eval
                and "optimizer" in checkpoint
                and "lr_scheduler" in checkpoint
                and "epoch" in checkpoint
            ):
                optimizer.load_state_dict(checkpoint["optimizer"])
                lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])
                args.start_epoch = checkpoint["epoch"] + 1
                if args.model_ema:
                    utils._load_checkpoint_for_ema(model_ema, checkpoint["model_ema"])
                if "scaler" in checkpoint:
                    loss_scaler.load_state_dict(checkpoint["scaler"])

            print(f"Resume from {args.resume} at epoch {args.start_epoch}.")

        if utils.is_main_process():
            writer = SummaryWriter(log_dir=str(output_dir))
        else:
            writer = None
        ##############################################
        # Test
        # if args.eval:
        #     breakpoint()
        # TODO: Match model's argument and optimizer's argument when resuming.
        test_stats = evaluate(
            data_loader_val,
            model,
            criterion,
            device,
            writer,
            -1,
            args.seed + 10000,
            args.valepisodes,
        )
        print(f"Accuracy of the network on dataset_val: {test_stats['acc1']:.3f}%")
        if args.output_dir and utils.is_main_process():
            test_stats["epoch"] = -1
            with (output_dir / "log.txt").open("a") as f:
                f.write(json.dumps(test_stats) + "\n")

        if args.eval:
            for source in data_loader_val.keys():
                # print(f"{source} loss: {test_stats[f'{source}_loss']}, acc1: {test_stats[f'{source}_acc1']}, acc5: {test_stats[f'{source}_acc5']}")
                print(f"{source} acc: {test_stats[f'{source}']}")
            return

        ##############################################
        # Training
        print(f"Start training for {args.epochs} epochs")
        start_time = time.time()
        max_accuracy = test_stats["acc1"]

        for epoch in range(args.start_epoch, args.epochs):
            train_stats = train_one_epoch(
                data_loader_train,
                model,
                criterion,
                optimizer,
                epoch,
                device,
                loss_scaler,
                args.fp16,
                args.clip_grad,
                model_ema,
                mixup_fn,
                writer,
                set_training_mode=False,  # TODO: may need eval mode for finetuning
                pretrain=args.pretrain,
            )

            lr_scheduler.step(epoch)

            test_stats = evaluate(
                data_loader_val,
                model,
                criterion,
                device,
                writer,
                epoch,
                seed=args.seed + 10000,
                ep=args.valepisodes,
            )

            log_stats = {
                **{f"train_{k}": v for k, v in train_stats.items()},
                **{f"test_{k}": v for k, v in test_stats.items()},
                "epoch": epoch,
                "n_parameters": n_parameters,
            }

            if args.output_dir:
                checkpoint_paths = [
                    output_dir / "checkpoint.pth",
                    output_dir / "best.pth",
                ]
                for checkpoint_path in checkpoint_paths:
                    state_dict = {
                        "model": model_without_ddp.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "lr_scheduler": lr_scheduler.state_dict(),
                        "epoch": epoch,
                        "model_ema": get_state_dict(model_ema)
                        if args.model_ema
                        else None,
                        "args": args,
                    }
                    if loss_scaler is not None:
                        state_dict["scalar"] = loss_scaler.state_dict()
                    utils.save_on_master(state_dict, checkpoint_path)

                    if test_stats["acc1"] <= max_accuracy:
                        break  # do not save best.pth

            print(f"Accuracy of the network on dataset_val: {test_stats['acc1']:.3f}%")
            max_accuracy = max(max_accuracy, test_stats["acc1"])
            print(f"Max accuracy: {max_accuracy:.2f}%")

            if args.output_dir and utils.is_main_process():
                log_stats["best_test_acc"] = max_accuracy
                with (output_dir / "log.txt").open("a") as f:
                    f.write(json.dumps(log_stats) + "\n")

        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print("Training time {}".format(total_time_str))

        if utils.is_main_process():
            writer.close()
            import tables

            tables.file._open_files.close_all()
    else:
        if args.optuna_study == "":
            study = optuna.create_study(direction="maximize")
        else:
            study = optuna.load_study(
                study_name=args.optuna_study, storage="sqlite:///bjk.db"
            )
        study.optimize(Objective(args, device), n_trials=args.optuna_n_trials)
        pprint.pprint(study.best_params)


class Objective:
    def __init__(self, args, device):
        self.args = args
        self.device = device

    def __call__(self, trial):
        args = self.args
        device = self.device
        loss_queue = deque(maxlen=10)
        acc_queue = deque(maxlen=10)
        train_loss_queue = deque(maxlen=10)
        train_acc_queue = deque(maxlen=10)
        output_dir = Path(args.output_dir + str(trial.number))

        # Re-assign the hyperparameters
        print_param = ""
        if "lr" in args.optuna:
            args.lr = trial.suggest_float("lr", 1e-4, 1e-1, log=True)
            print_param += f"lr: {args.lr}, "
        if "weight_decay" in args.optuna:
            args.weight_decay = trial.suggest_float(
                "weight_decay", 1e-5, 1e-1, log=True
            )
            print_param += f"weight_decay: {args.weight_decay}, "
        if "decay_epochs" in args.optuna:
            args.decay_epochs = trial.suggest_int("decay_epochs", 10, 100)
            print_param += f"decay_epochs: {args.decay_epochs}, "
        if "decay_rate" in args.optuna:
            args.decay_rate = trial.suggest_float("decay_rate", 0.1, 1.0, log=True)
            print_param += f"decay_rate: {args.decay_rate}, "
        if "inner_steps" in args.optuna:
            args.inner_steps = trial.suggest_int("inner_steps", 1, 5)
            print_param += f"inner_steps: {args.inner_steps}, "
        if "inner_lr" in args.optuna:
            args.inner_lr = trial.suggest_float("inner_lr", 1e-5, 1e-1, log=True)
            print_param += f"inner_lr: {args.inner_lr}, "
        if "contrastive_lr" in args.optuna:
            args.contrastive_lr = trial.suggest_float(
                "contrastive_lr", 1e-5, 1e-1, log=True
            )
            print_param += f"contrastive_lr: {args.contrastive_lr}, "
        if "contrastive_steps" in args.optuna:
            args.contrastive_steps = trial.suggest_int("contrastive_steps", 3, 10)
            print_param += f"contrastive_steps: {args.contrastive_steps}, "
        if "warmup_epochs" in args.optuna:
            args.warmup_epochs = trial.suggest_int("warmup_epochs", 0, 30, step=5)
            print_param += f"warmup_epochs: {args.warmup_epochs}, "
        # if 'cooldown_epochs' in args.optuna:
        #     args.cooldown_epochs = trial.suggest_int('cooldown_epochs', 0, 30, step=5)
        #     print_param += f'cooldown_epochs: {args.cooldown_epochs}, '

        if utils.is_main_process():
            try:
                output_dir.mkdir(parents=True, exist_ok=False)
            except FileExistsError:
                input(
                    "Warning: Output directory already exists and is not empty. Press Enter to continue..."
                )
                output_dir.mkdir(parents=True, exist_ok=True)
            with (output_dir / "log.txt").open("a") as f:
                f.write(" ".join(sys.argv) + "\n")
                # Write rewritten hyperparameters
                f.write(print_param + "\n")
                print(print_param)
        ##############################################
        # Data loaders
        num_tasks = utils.get_world_size()
        global_rank = utils.get_rank()
        data_loader_train, data_loader_val = get_loaders(
            args, num_tasks, global_rank, transform_type=args.transform_type
        )
        ##############################################
        # Mixup regularization (by default OFF)
        mixup_fn = None
        mixup_active = (
            args.mixup > 0 or args.cutmix > 0.0 or args.cutmix_minmax is not None
        )
        if mixup_active:
            mixup_fn = Mixup(
                mixup_alpha=args.mixup,
                cutmix_alpha=args.cutmix,
                cutmix_minmax=args.cutmix_minmax,
                prob=args.mixup_prob,
                switch_prob=args.mixup_switch_prob,
                mode=args.mixup_mode,
                label_smoothing=args.smoothing,
                num_classes=args.nClsEpisode,
            )

        ##############################################
        # Model
        print(f"Creating model: ProtoNet {args.arch}")

        model = get_model(args)
        model = model.to(device)

        model_ema = None  # (by default OFF)
        if args.model_ema:
            # Important to create EMA model after cuda(), DP wrapper, and AMP but before SyncBN and DDP wrapper
            model_ema = ModelEma(
                model,
                decay=args.model_ema_decay,
                device="cpu" if args.model_ema_force_cpu else "",
                resume="",
            )

        model_without_ddp = model
        if args.distributed:
            model = torch.nn.parallel.DistributedDataParallel(
                model, device_ids=[args.gpu], find_unused_parameters=args.unused_params
            )
            model_without_ddp = model.module
        n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print("number of params:", n_parameters)

        ##############################################
        # Optimizer & scheduler & criterion
        if args.fp16:
            scale = 1 / 8  # the default lr is for 8 GPUs
            linear_scaled_lr = args.lr * utils.get_world_size() * scale
            args.lr = linear_scaled_lr

        loss_scaler = NativeScaler() if args.fp16 else None

        # optimizer = create_optimizer(args, model_without_ddp)
        optimizer = torch.optim.SGD(
            [p for p in model_without_ddp.parameters() if p.requires_grad],
            args.lr,
            momentum=args.momentum,
            weight_decay=args.weight_decay,  # no weight decay for fine-tuning
        )

        lr_scheduler, _ = create_scheduler(args, optimizer)

        if args.mixup > 0.0:
            # smoothing is handled with mixup label transform
            criterion = SoftTargetCrossEntropy()
        elif args.smoothing:
            criterion = LabelSmoothingCrossEntropy(smoothing=args.smoothing)
        else:
            criterion = torch.nn.CrossEntropyLoss()

        ##############################################
        # Resume training from ckpt (model, optimizer, lr_scheduler, epoch, model_ema, scaler)
        if args.resume:
            if args.resume.startswith("https"):
                checkpoint = torch.hub.load_state_dict_from_url(
                    args.resume, map_location="cpu", check_hash=True
                )
            else:
                checkpoint = torch.load(args.resume, map_location="cpu")

            model_without_ddp.load_state_dict(checkpoint["model"])

            if (
                not args.eval
                and "optimizer" in checkpoint
                and "lr_scheduler" in checkpoint
                and "epoch" in checkpoint
            ):
                optimizer.load_state_dict(checkpoint["optimizer"])
                lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])
                args.start_epoch = checkpoint["epoch"] + 1
                if args.model_ema:
                    utils._load_checkpoint_for_ema(model_ema, checkpoint["model_ema"])
                if "scaler" in checkpoint:
                    loss_scaler.load_state_dict(checkpoint["scaler"])

            print(f"Resume from {args.resume} at epoch {args.start_epoch}.")

        if utils.is_main_process():
            writer = SummaryWriter(log_dir=str(output_dir))
        else:
            writer = None
        ##############################################
        # # Test
        # test_stats = evaluate(data_loader_val, model, criterion, device, writer, -1, args.seed+10000, args.valepisodes)
        # print(f"Accuracy of the network on dataset_val: {test_stats['acc1']:.1f}%")
        # if args.output_dir and utils.is_main_process():
        #     test_stats['epoch'] = -1
        #     with (output_dir / "log.txt").open("a") as f:
        #         f.write(json.dumps(test_stats) + "\n")

        # if args.eval:
        #     return

        ##############################################
        # Training

        print(f"Start training for {args.epochs} epochs")
        start_time = time.time()
        max_accuracy = -1
        min_loss = 100000
        early_stop_count = 0

        for epoch in range(args.start_epoch, args.epochs):
            train_stats = train_one_epoch(
                data_loader_train,
                model,
                criterion,
                optimizer,
                epoch,
                device,
                loss_scaler,
                args.fp16,
                args.clip_grad,
                model_ema,
                mixup_fn,
                writer,
                set_training_mode=False,  # TODO: may need eval mode for finetuning
                pretrain=args.pretrain,
            )

            lr_scheduler.step(epoch)

            test_stats = evaluate(
                data_loader_val,
                model,
                criterion,
                device,
                writer,
                epoch,
                seed=args.seed + 10000,
                ep=args.valepisodes,
            )

            log_stats = {
                **{f"train_{k}": v for k, v in train_stats.items()},
                **{f"test_{k}": v for k, v in test_stats.items()},
                "epoch": epoch,
                "n_parameters": n_parameters,
            }

            if args.output_dir:
                checkpoint_paths = [
                    output_dir / "checkpoint.pth",
                    output_dir / "best.pth",
                ]
                for checkpoint_path in checkpoint_paths:
                    state_dict = {
                        "model": model_without_ddp.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "lr_scheduler": lr_scheduler.state_dict(),
                        "epoch": epoch,
                        "model_ema": get_state_dict(model_ema)
                        if args.model_ema
                        else None,
                        "args": args,
                    }
                    if loss_scaler is not None:
                        state_dict["scalar"] = loss_scaler.state_dict()
                    utils.save_on_master(state_dict, checkpoint_path)

                    if test_stats["acc1"] <= max_accuracy:
                        break  # do not save best.pth

            min_loss = min(min_loss, test_stats["loss"])

            print(f"Accuracy of the network on dataset_val: {test_stats['acc1']:.3f}%")
            max_accuracy = max(max_accuracy, test_stats["acc1"])
            print(f"Max accuracy: {max_accuracy:.2f}%")

            loss_queue.append(test_stats["loss"])
            acc_queue.append(test_stats["acc1"])

            # score = np.mean(acc_queue) / (np.mean(loss_queue) + 1e-5)
            score = np.mean(acc_queue)

            if args.output_dir and utils.is_main_process():
                log_stats["best_test_acc"] = max_accuracy
                with (output_dir / "log.txt").open("a") as f:
                    f.write(json.dumps(log_stats) + "\n")

            # Early stopping with patience 5, or stop immediately if loss is too high.
            if np.isnan(test_stats["loss"]):
                return score
            if len(loss_queue) == loss_queue.maxlen:
                if np.mean(loss_queue) > min_loss * 1.3:
                    return score
                if np.mean(acc_queue) < max_accuracy * 0.95:
                    early_stop_count += 1
                    if early_stop_count >= 5:
                        return score
                else:
                    early_stop_count = 0
            if len(train_loss_queue) == train_loss_queue.maxlen:
                if np.mean(train_loss_queue) == 0 or np.mean(train_acc_queue) == 0:
                    return score

        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print("Training time {}".format(total_time_str))

        if utils.is_main_process():
            writer.close()
            import tables

            tables.file._open_files.close_all()

        return score


if __name__ == "__main__":
    parser = get_args_parser()
    args = parser.parse_args()

    main(args)
