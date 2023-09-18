import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import learn2learn as l2l
import math
from itertools import chain
from copy import deepcopy
import torch

from torch import optim
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import os

from .learner import Learner
from copy import deepcopy
from torchvision.transforms import transforms
from torchvision.transforms import functional as TF
from pytorch_metric_learning import losses


class Rotate:
    def __init__(self, angles: list):
        self.angles = angles

    def __call__(self, img):
        angle = random.choice(self.angles)
        return TF.rotate(img, angle)


def transform(img):
    """ """
    original_size = img.size()
    _transform = transforms.Compose(
        [
            transforms.RandomCrop(84),
            transforms.Resize(original_size[2]),
            transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.4),
        ]
    )

    return _transform(img)


class Meta_mini(nn.Module):
    def __init__(
        self,
        meta_lr,
        inner_lr,
        inner_steps,
        inner_eval_steps,
        contrastive_lr,
        contrastive_steps,
        first_order,
        config,
    ):
        super(Meta_mini, self).__init__()

        self.update_lr = inner_lr
        self.meta_lr = meta_lr
        self.update_step = inner_steps
        self.update_step_test = inner_eval_steps
        self.contrastive_lr = contrastive_lr
        self.contrastive_steps = contrastive_steps
        self.first_order = first_order
        if first_order:
            self.forward = self.forward_FOMAML
        else:
            self.forward = self.forward_SOMAML
        self.linear_shape = 36
        self.net = Learner(config, 3, 84)
        self.meta_optim = optim.Adam(self.net.parameters(), lr=self.meta_lr)

    def load_model(self, save_path, epoch):
        path = os.path.join(save_path, "E{}S0.pt".format(epoch))
        checkpoint = torch.load(path)
        self.net.load_state_dict(checkpoint["model_state_dict"])
        self.meta_optim.load_state_dict(checkpoint["optimizer_state_dict"])
        self.net.train()

    def save_model(self, save_path, epoch, step):
        path = os.path.join(save_path, "E{}S{}.pt".format(epoch, step))
        torch.save(
            {
                "epoch": epoch,
                "step": step,
                "model_state_dict": self.net.state_dict(),
                "optimizer_state_dict": self.meta_optim.state_dict(),
            },
            path,
        )

    def clip_grad_by_norm_(self, grad, max_norm):
        """
        in-place gradient clipping.
        :param grad: list of gradients
        :param max_norm: maximum norm allowable
        :return:
        """
        total_norm = 0
        counter = 0
        for g in grad:
            param_norm = g.data.norm(2)
            total_norm += param_norm.item() ** 2
            counter += 1
        total_norm = total_norm ** (1.0 / 2)

        clip_coef = max_norm / (total_norm + 1e-6)
        if clip_coef < 1:
            for g in grad:
                g.data.mul_(clip_coef)

        return total_norm / counter

    def set_last_layer_to_zero(self):
        for p1, p2 in self.net.named_parameters():
            if p1 == "vars.16" or p1 == "vars.17":
                p2.data = torch.zeros_like(p2.data)

    def set_last_layer_variance(self, var):
        for p1, p2 in self.net.named_parameters():
            if p1 == "vars.16" or p1 == "vars.17":
                p2.data = p2.data * var
    
    def get_fc_configs(self, y_qry):
        if isinstance(y_qry, int):
            # Contrastive Learning y_qry actually doesn't stand for
            # query labels, but is the number of tasks
            n_ways = [32]*y_qry
        else:
            n_ways = list(map(lambda x: max(x).item() + 1, y_qry))
            
        return [[
            ("linear", [n_way, 32 * self.linear_shape]),
        ] for n_way in n_ways]

        # Don't need meta optim for last layer
        # self.opt_fc = optim.Adam(self.net.named_parameters[], lr=self.meta_lr)


    def forward_FOMAML(self, x_spt, y_spt, x_qry, y_qry):
        task_num, setsz, c_, h, w = x_spt.size()
        querysz = x_qry.size(1)
        contrastive_configs = self.get_fc_configs(len(y_qry))
        fc_configs = self.get_fc_configs(y_qry)

        # self.meta_optim.zero_grad()
        # record per step
        losses_q = [0 for _ in range(self.update_step + 1)]
        corrects = [0 for _ in range(self.update_step + 1)]

        # Variables for contrastive learning
        loss_fn = losses.SelfSupervisedLoss(losses.NTXentLoss())
        x_spt_aug = [transform(img) for img in x_spt]
        fast_weights_all_tasks = []
        # Contrastive Learning
        for i in range(task_num):
            if 'vars.16' in list(zip(*list(self.net.named_parameters())))[0]:
                self.net.pop()
            self.net.append(contrastive_configs[i])
            self.net.cuda()
            # self.set_last_layer_to_zero() # Problematic

            fast_weights = self.net.parameters()
            # print('----')
            for k in range(self.contrastive_steps):
                # 1. run the i-th task and compute loss for k=1~K-1
                keys = self.net(x_spt[i], fast_weights, bn_training=True)
                queries = self.net(x_spt_aug[i], fast_weights, bn_training=True)
                loss = loss_fn(keys, queries)
                # print('Contrastive Loss: {}'.format(loss.item()))
                # 2. compute grad on theta_pi
                grad = torch.autograd.grad(loss, fast_weights)
                # 3. theta_pi = theta_pi - train_lr * grad
                fast_weights = list(
                    map(lambda p: p[1] - self.contrastive_lr * p[0], zip(grad, fast_weights))
                )
            fast_weights_all_tasks.append(fast_weights)

        # Get mean of fast_weights
        contrastive_initialized_weights = list(map(lambda x: torch.stack(x).mean(0), zip(*fast_weights_all_tasks)))
        # fast_weights = self.net.parameters()
        for i in range(task_num):
            if 'vars.16' in list(zip(*list(self.net.named_parameters())))[0]:
                self.net.pop()
            self.net.append(fc_configs[i])
            self.net.cuda()
            self.set_last_layer_to_zero()

            # fast_weights = self.net.parameters()
            fast_weights = contrastive_initialized_weights[:-2] + list(self.net.parameters())[-2:]

            # 1. run the i-th task and compute loss for k=0
            logits = self.net(x_spt[i], vars=None, bn_training=True)
            loss = F.cross_entropy(logits, y_spt[i])
            grad = torch.autograd.grad(loss, self.net.parameters())
            fast_weights = list(
                map(lambda p: p[1] - self.update_lr * p[0], zip(grad, fast_weights))
            )

            # this is the loss and accuracy before first update
            with torch.no_grad():
                # [setsz, nway]
                logits_q = self.net(x_qry[i], self.net.parameters(), bn_training=True)
                loss_q = F.cross_entropy(logits_q, y_qry[i])
                losses_q[0] += loss_q

                pred_q = F.softmax(logits_q, dim=1).argmax(dim=1)
                correct = torch.eq(pred_q, y_qry[i]).sum().item()
                corrects[0] = corrects[0] + correct

            for k in range(1, self.update_step):
                # 1. run the i-th task and compute loss for k=1~K-1
                logits = self.net(x_spt[i], fast_weights, bn_training=True)
                loss = F.cross_entropy(logits, y_spt[i])
                # 2. compute grad on theta_pi
                grad = torch.autograd.grad(loss, fast_weights)
                # 3. theta_pi = theta_pi - train_lr * grad
                fast_weights = list(
                    map(lambda p: p[1] - self.update_lr * p[0], zip(grad, fast_weights))
                )

                logits_q = self.net(x_qry[i], fast_weights, bn_training=True)
                # loss_q will be overwritten and just keep the loss_q on last update step.
                loss_q = F.cross_entropy(logits_q, y_qry[i])
                losses_q[k + 1] += loss_q

                with torch.no_grad():
                    pred_q = F.softmax(logits_q, dim=1).argmax(dim=1)
                    correct = torch.eq(pred_q, y_qry[i]).sum().item()  # convert to numpy
                    corrects[k + 1] = corrects[k + 1] + correct
                loss_q.backward(retain_graph=True)
                # loss_q.backward(retain_graph=k!=self.update_step-1)
            

        # end of all tasks
        # sum over all losses on query set across all tasks
        loss_q = losses_q[-1] / task_num

        # self.meta_optim.step()
        accs = np.array(corrects) / (querysz * task_num)
        if 'vars.16' in list(zip(*list(self.net.named_parameters())))[0]:
            self.net.pop()
        return loss_q.item(), accs[-1]

    def forward_SOMAML(self, x_spt, y_spt, x_qry, y_qry):
        """

        :param x_spt:   [b, setsz, c_, h, w]
        :param y_spt:   [b, setsz]
        :param x_qry:   [b, querysz, c_, h, w]
        :param y_qry:   [b, querysz]
        :return:
        """
        task_num, setsz, c_, h, w = x_spt.size()
        querysz = x_qry.size(1)

        # record per step
        losses_q = [0 for _ in range(self.update_step + 1)]
        losses_q_final = list()
        corrects = [0 for _ in range(self.update_step + 1)]

        for i in range(task_num):
            logits = self.net(x_spt[i], vars=None, bn_training=True)
            loss = F.cross_entropy(logits, y_spt[i])
            grad = torch.autograd.grad(
                loss, self.net.parameters(), retain_graph=True, create_graph=True
            )
            fast_weights = list(
                map(lambda p: p[1] - self.update_lr * p[0], zip(grad, self.net.parameters()))
            )

            # this is the loss and accuracy before first update
            with torch.no_grad():
                # [setsz, nway]
                logits_q = self.net(x_qry[i], self.net.parameters(), bn_training=True)
                loss_q = F.cross_entropy(logits_q, y_qry[i])
                losses_q[0] += loss_q

                pred_q = F.softmax(logits_q, dim=1).argmax(dim=1)
                correct = torch.eq(pred_q, y_qry[i]).sum().item()
                corrects[0] = corrects[0] + correct

            # this is the loss and accuracy after the first update
            with torch.no_grad():
                # [setsz, nway]
                logits_q = self.net(x_qry[i], fast_weights, bn_training=True)
                loss_q = F.cross_entropy(logits_q, y_qry[i])
                losses_q[1] += loss_q
                # [setsz]
                pred_q = F.softmax(logits_q, dim=1).argmax(dim=1)
                correct = torch.eq(pred_q, y_qry[i]).sum().item()
                corrects[1] = corrects[1] + correct

            for k in range(1, self.update_step):
                logits = self.net(x_spt[i], fast_weights, bn_training=True)
                loss = F.cross_entropy(logits, y_spt[i])
                grad = torch.autograd.grad(loss, fast_weights, retain_graph=True, create_graph=True)
                fast_weights = list(
                    map(lambda p: p[1] - self.update_lr * p[0], zip(grad, fast_weights))
                )

                logits_q = self.net(x_qry[i], fast_weights, bn_training=True)
                # loss_q will be overwritten and just keep the loss_q on last update step.
                loss_q = F.cross_entropy(logits_q, y_qry[i])
                losses_q[k + 1] += loss_q

                if k == (self.update_step - 1):
                    loss_q = F.cross_entropy(logits_q, y_qry[i])
                    loss_q.backward(retain_graph=True)
                    losses_q_final.append(loss_q)

                with torch.no_grad():
                    pred_q = F.softmax(logits_q, dim=1).argmax(dim=1)
                    correct = torch.eq(pred_q, y_qry[i]).sum().item()  # convert to numpy
                    corrects[k + 1] = corrects[k + 1] + correct

        self.meta_optim.zero_grad()
        meta_batch_loss = torch.stack(losses_q_final).mean()
        meta_batch_loss.backward()
        self.meta_optim.step()
        accs = np.array(corrects) / (querysz * task_num)

        return accs
    
    def evaluate(self, x_spt, y_spt, x_qry, y_qry):
        task_num, setsz, c_, h, w = x_spt.size()
        querysz = x_qry.size(1)

        losses_q = [0 for _ in range(self.update_step_test + 1)]
        corrects = [0 for _ in range(self.update_step_test + 1)]
        contrastive_configs = self.get_fc_configs(len(y_qry))
        fc_configs = self.get_fc_configs(y_qry)

        net = deepcopy(self.net)
        # record per step
        losses_q = [0 for _ in range(self.update_step_test + 1)]
        corrects = [0 for _ in range(self.update_step_test + 1)]

        # Variables for contrastive learning
        loss_fn = losses.SelfSupervisedLoss(losses.NTXentLoss())
        x_spt_aug = [transform(img) for img in x_spt]
        fast_weights_all_tasks = []
        net.train()
        for i in range(task_num):    
            # Contrastive Learning
            if 'vars.16' in list(zip(*list(net.named_parameters())))[0]:
                net.pop()

            net.append(contrastive_configs[i])
            net.cuda()
            # self.set_last_layer_to_zero()

            fast_weights = net.parameters()
            for k in range(self.update_step_test):
                # 1. run the i-th task and compute loss for k=1~K-1
                keys = net(x_spt[i], fast_weights, bn_training=True)
                queries = net(x_spt_aug[i], fast_weights, bn_training=True)
                loss = loss_fn(keys, queries)
                # 2. compute grad on theta_pi
                grad = torch.autograd.grad(loss, fast_weights)
                # 3. theta_pi = theta_pi - train_lr * grad
                fast_weights = list(
                    map(lambda p: p[1] - self.update_lr * p[0], zip(grad, fast_weights))
                )
            fast_weights_all_tasks.append(fast_weights)

        # Get mean of fast_weights
        contrastive_initialized_weights = list(map(lambda x: torch.stack(x).mean(0), zip(*fast_weights_all_tasks)))
        # fast_weights = net.parameters()
        for i in range(task_num):
            if 'vars.16' in list(zip(*list(net.named_parameters())))[0]:
                net.pop()
            net.append(fc_configs[i])
            net.cuda()
            self.set_last_layer_to_zero()

            # fast_weights = net.parameters()
            fast_weights = contrastive_initialized_weights[:-2] + list(net.parameters())[-2:]

            # 1. run the i-th task and compute loss for k=0
            logits = net(x_spt[i], vars=None, bn_training=True)
            loss = F.cross_entropy(logits, y_spt[i])
            grad = torch.autograd.grad(loss, net.parameters())
            fast_weights = list(
                map(lambda p: p[1] - self.update_lr * p[0], zip(grad, fast_weights))
            )

            # this is the loss and accuracy before first update
            with torch.no_grad():
                # [setsz, nway]
                logits_q = net(x_qry[i], net.parameters(), bn_training=True)
                loss_q = F.cross_entropy(logits_q, y_qry[i])
                losses_q[0] += loss_q

                pred_q = F.softmax(logits_q, dim=1).argmax(dim=1)
                correct = torch.eq(pred_q, y_qry[i]).sum().item()
                corrects[0] = corrects[0] + correct

            for k in range(1, self.update_step_test):
                # 1. run the i-th task and compute loss for k=1~K-1
                logits = net(x_spt[i], fast_weights, bn_training=True)
                loss = F.cross_entropy(logits, y_spt[i])
                # 2. compute grad on theta_pi
                grad = torch.autograd.grad(loss, fast_weights)
                # 3. theta_pi = theta_pi - train_lr * grad
                fast_weights = list(
                    map(lambda p: p[1] - self.update_lr * p[0], zip(grad, fast_weights))
                )

                logits_q = net(x_qry[i], fast_weights, bn_training=True)
                # loss_q will be overwritten and just keep the loss_q on last update step.
                loss_q = F.cross_entropy(logits_q, y_qry[i])
                losses_q[k + 1] += loss_q

                with torch.no_grad():
                    pred_q = F.softmax(logits_q, dim=1).argmax(dim=1)
                    correct = torch.eq(pred_q, y_qry[i]).sum().item()  # convert to numpy
                    corrects[k + 1] = corrects[k + 1] + correct

        # end of all tasks
        # sum over all losses on query set across all tasks
        loss_q = losses_q[-1] / task_num

        accs = np.array(corrects) / (querysz * task_num)
        return loss_q, accs[-1]

    def finetunning(self, x_spt, y_spt, x_qry, y_qry):
        """
        :param x_spt:   [setsz, c_, h, w]
        :param y_spt:   [setsz]
        :param x_qry:   [querysz, c_, h, w]
        :param y_qry:   [querysz]
        :return:
        """
        assert len(x_spt.shape) == 4

        querysz = x_qry.size(0)

        corrects = [0 for _ in range(self.update_step_test + 1)]

        # in order to not ruin the state of running_mean/variance and bn_weight/bias
        # we finetunning on the copied model instead of self.net
        net = deepcopy(self.net)

        # 1. run the i-th task and compute loss for k=0
        logits = net(x_spt)
        loss = F.cross_entropy(logits, y_spt)
        grad = torch.autograd.grad(loss, net.parameters())
        fast_weights = list(
            map(lambda p: p[1] - self.update_lr * p[0], zip(grad, net.parameters()))
        )

        # this is the loss and accuracy before first update
        with torch.no_grad():
            # [setsz, nway]
            logits_q = net(x_qry, net.parameters(), bn_training=True)
            # [setsz]
            pred_q = F.softmax(logits_q, dim=1).argmax(dim=1)
            # scalar
            correct = torch.eq(pred_q, y_qry).sum().item()
            corrects[0] = corrects[0] + correct

        # this is the loss and accuracy after the first update
        with torch.no_grad():
            # [setsz, nway]
            logits_q = net(x_qry, fast_weights, bn_training=True)
            # [setsz]
            pred_q = F.softmax(logits_q, dim=1).argmax(dim=1)
            # scalar
            correct = torch.eq(pred_q, y_qry).sum().item()
            corrects[1] = corrects[1] + correct

        for k in range(1, self.update_step_test):
            # 1. run the i-th task and compute loss for k=1~K-1
            logits = net(x_spt, fast_weights, bn_training=True)
            loss = F.cross_entropy(logits, y_spt)
            # 2. compute grad on theta_pi
            grad = torch.autograd.grad(loss, fast_weights)
            # 3. theta_pi = theta_pi - train_lr * grad
            fast_weights = list(
                map(lambda p: p[1] - self.update_lr * p[0], zip(grad, fast_weights))
            )

            logits_q = net(x_qry, fast_weights, bn_training=True)
            # loss_q will be overwritten and just keep the loss_q on last update step.
            loss_q = F.cross_entropy(logits_q, y_qry)

            with torch.no_grad():
                pred_q = F.softmax(logits_q, dim=1).argmax(dim=1)
                correct = torch.eq(pred_q, y_qry).sum().item()  # convert to numpy
                corrects[k + 1] = corrects[k + 1] + correct
        del net
        accs = np.array(corrects) / querysz
        return accs

    def set_net_last_layer_to_zero(self, net):
        for p1, p2 in net.named_parameters():
            if p1 == "vars.16" or p1 == "vars.17":
                p2.data = torch.zeros_like(p2.data)

    def finetunning_zero(self, x_spt, y_spt, x_qry, y_qry):
        assert len(x_spt.shape) == 4
        querysz = x_qry.size(0)

        corrects = [0 for _ in range(self.update_step_test + 1)]

        net = deepcopy(self.net)

        self.set_net_last_layer_to_zero(net)

        logits = net(x_spt)
        loss = F.cross_entropy(logits, y_spt)
        grad = torch.autograd.grad(loss, net.parameters())
        fast_weights = list(
            map(lambda p: p[1] - self.update_lr * p[0], zip(grad, net.parameters()))
        )

        with torch.no_grad():
            # [setsz, nway]
            logits_q = net(x_qry, net.parameters(), bn_training=True)
            pred_q = F.softmax(logits_q, dim=1).argmax(dim=1)
            correct = torch.eq(pred_q, y_qry).sum().item()
            corrects[0] = corrects[0] + correct

        with torch.no_grad():
            logits_q = net(x_qry, fast_weights, bn_training=True)
            pred_q = F.softmax(logits_q, dim=1).argmax(dim=1)
            correct = torch.eq(pred_q, y_qry).sum().item()
            corrects[1] = corrects[1] + correct

        for k in range(1, self.update_step_test):
            logits = net(x_spt, fast_weights, bn_training=True)
            loss = F.cross_entropy(logits, y_spt)
            grad = torch.autograd.grad(loss, fast_weights)
            fast_weights = list(
                map(lambda p: p[1] - self.update_lr * p[0], zip(grad, fast_weights))
            )

            logits_q = net(x_qry, fast_weights, bn_training=True)
            # loss_q will be overwritten and just keep the loss_q on last update step.
            loss_q = F.cross_entropy(logits_q, y_qry)

            with torch.no_grad():
                pred_q = F.softmax(logits_q, dim=1).argmax(dim=1)
                correct = torch.eq(pred_q, y_qry).sum().item()  # convert to numpy
                corrects[k + 1] = corrects[k + 1] + correct
        del net
        accs = np.array(corrects) / querysz
        return accs

    def get_feature(self, x, y):
        """
        :param x_spt:   [setsz, c_, h, w]
        :param y_spt:   [setsz]
        :param x_qry:   [querysz, c_, h, w]
        :param y_qry:   [querysz]
        :return:
        """
        assert len(x.shape) == 4

        querysz = x.size(0)
        corrects = [0 for _ in range(self.update_step_test + 1)]

        # in order to not ruin the state of running_mean/variance and bn_weight/bias
        # we finetunning on the copied model instead of self.net
        net = deepcopy(self.net)

        # this is the loss and accuracy before first update
        with torch.no_grad():
            features = net.get_feature(x, net.parameters(), bn_training=True)
            logits_q = net(x, net.parameters(), bn_training=True)
            pred_q = F.softmax(logits_q, dim=1).argmax(dim=1)
            correct = torch.eq(pred_q, y).sum().item()
            corrects[0] = corrects[0] + correct

        return features, logits_q, pred_q, correct
