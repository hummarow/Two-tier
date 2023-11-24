import torch
from torch import nn
from torch.nn import functional as F
import numpy as np


class ParamList(nn.ParameterList):
    def __init__(self, *args, **kwargs):
        super(ParamList, self).__init__(*args, **kwargs)
        self.idx = 0

    def make_valid_checker(self):
        self.bool_list = [False] * len(self)

    def pop(self, garbage):
        if self.idx == len(self):
            return None
        p = self[self.idx]
        self.bool_list[self.idx] = True
        self.idx += 1
        return p

    def valid_check(self, stop=None):
        if stop is None:
            stop = len(self)
        return np.all(self.bool_list[:stop])

    def initialize(self):
        self.bool_list = [False] * len(self)
        self.idx = 0


class Learner(nn.Module):
    def __init__(self, config, imgc, imgsz):
        """

        :param config: network config file, type:list of (string, list)
        :param imgc: 1 or 3
        :param imgsz:  28 or 84
        """
        super(Learner, self).__init__()

        # # this dict contains all tensors needed to be optimized
        # self.vars = nn.ParameterList()
        # # running_mean and running_var
        # self.vars_bn = nn.ParameterList()
        self.vars = ParamList()
        self.vars_bn = ParamList()

        self.config = []
        self.identity_size = None
        self.append(config)

    def append(self, config, zero=False):
        def extract_basicblock(ch_out, ch_in, stride=1, expansion=1):
            return [
                ("conv2d", [ch_out, ch_in, 3, 3, stride, 1]),
                ("bn", [ch_out]),
                ("relu", [True]),
                ("conv2d", [ch_out, ch_out, 3, 3, 1, 1]),
                ("bn", [ch_out]),
            ]

        # Extract the basic block in the config list.
        i = 0
        while i < len(config):
            if config[i][0] == "basicblock":
                ch_out, ch_in, stride, expansion = config[i][1]
                basicblock = extract_basicblock(ch_out, ch_in, stride, expansion)
                config = config[:i] + basicblock + config[i + 1 :]
            i += 1

        for i, (name, param) in enumerate(config):
            if name == "conv2d":
                # [ch_out, ch_in, kernelsz, kernelsz]
                w = nn.Parameter(torch.ones(*param[:4]))
                # gain=1 according to cbfin's implementation
                torch.nn.init.kaiming_normal_(w)
                self.vars.append(w)
                # [ch_out]
                self.vars.append(nn.Parameter(torch.zeros(param[0])))

            elif name == "convt2d":
                # [ch_in, ch_out, kernelsz, kernelsz, stride, padding]
                w = nn.Parameter(torch.ones(*param[:4]))
                # gain=1 according to cbfin's implementation
                torch.nn.init.kaiming_normal_(w)
                self.vars.append(w)
                # [ch_in, ch_out]
                self.vars.append(nn.Parameter(torch.zeros(param[1])))

            elif name == "linear":
                # [ch_out, ch_in]
                w = nn.Parameter(torch.ones(*param))
                # gain=1 according to cbfinn's implementation
                if zero:
                    torch.nn.init.zeros_(w)
                else:
                    torch.nn.init.kaiming_normal_(w)
                self.vars.append(w)
                # [ch_out]
                self.vars.append(nn.Parameter(torch.zeros(param[0])))

            elif name == "bn":
                # [ch_out]
                w = nn.Parameter(torch.ones(param[0]))
                self.vars.append(w)
                # [ch_out]
                self.vars.append(nn.Parameter(torch.zeros(param[0])))

                # must set requires_grad=False
                running_mean = nn.Parameter(torch.zeros(param[0]), requires_grad=False)
                running_var = nn.Parameter(torch.ones(param[0]), requires_grad=False)
                self.vars_bn.extend([running_mean, running_var])
            elif name == "identity_in":
                # Record the size of the input.
                self.identity_size = None if not param else param[0]
            elif name == "identity_out":
                assert bool(self.identity_size) == bool(param)
                if self.identity_size is None:
                    # If the identity_size is None, then the identity_in is not called.
                    # So we just skip this layer.
                    continue
                # [ch_out, ch_in, kernelsz, kernelsz]
                w = nn.Parameter(torch.ones([param[0], self.identity_size, 1, 1]))
                torch.nn.init.kaiming_normal_(w)
                self.vars.append(w)
                # [ch_out]
                self.vars.append(nn.Parameter(torch.zeros(param[0])))
                self.identity_size = None
            elif name in [
                "tanh",
                "relu",
                "upsample",
                "avg_pool2d",
                "max_pool2d",
                "flatten",
                "reshape",
                "leakyrelu",
                "sigmoid",
            ]:
                continue
            else:
                breakpoint()
                raise NotImplementedError

        self.config.extend(config)
        self.vars.make_valid_checker()
        self.vars_bn.make_valid_checker()

    def pop(self):
        layer_old, vars_old = self.config[-1], self.vars[-2:]
        layer = self.config[-1]
        if layer[0] == "bn":
            vars_bn_old = self.vars_bn[-2:]
            self.vars_bn = self.vars_bn[:-2]
        else:
            vars_bn_old = None
        self.vars = self.vars[:-2]

        self.config = self.config[:-1]

        return layer_old, vars_old, vars_bn_old

    def forward(self, x, vars=None, bn_training=True):
        x, _ = self.forward_feature(x, vars, bn_training)
        return x

    def forward_feature(self, x, vars=None, bn_training=True):
        if vars is None:
            vars = self.vars

        idx = 0
        bn_idx = 0
        x_flatten = None
        for i, (name, param) in enumerate(self.config):
            if name == "conv2d":
                w, b = vars[idx], vars[idx + 1]
                # w = vars.pop(0)
                # b = vars.pop(0)
                # remember to keep synchrozied of forward_encoder and forward_decoder!
                x = F.conv2d(x, w, b, stride=param[4], padding=param[5])
                idx += 2
            elif name == "linear":
                x_flatten = x
                w, b = vars[idx], vars[idx + 1]
                # w = vars.pop(0)
                # b = vars.pop(0)
                x = F.linear(x, w, b)
                idx += 2
            elif name == "bn":
                w, b = vars[idx], vars[idx + 1]
                # w = vars.pop(0)
                # b = vars.pop(0)
                if len(w.shape) == 2:
                    breakpoint()
                running_mean, running_var = (
                    self.vars_bn[bn_idx],
                    self.vars_bn[bn_idx + 1],
                )
                # running_mean = self.vars_bn.pop(0)
                # running_var = self.vars_bn.pop(0)
                x = F.batch_norm(
                    x, running_mean, running_var, weight=w, bias=b, training=bn_training
                )
                idx += 2
                bn_idx += 2
            elif name == "flatten":
                # print(x.shape)
                # x = x.view(x.size(0), -1)
                x = x.reshape(x.size(0), -1)
            elif name == "reshape":
                # [b, 8] => [b, 2, 2, 2]
                x = x.view(x.size(0), *param)
            elif name == "relu":
                x = F.relu(x, inplace=param[0])
            elif name == "leakyrelu":
                x = F.leaky_relu(x, negative_slope=param[0], inplace=param[1])
            elif name == "tanh":
                x = F.tanh(x)
            elif name == "sigmoid":
                x = torch.sigmoid(x)
            elif name == "upsample":
                x = F.upsample_nearest(x, scale_factor=param[0])
            elif name == "max_pool2d":
                x = F.max_pool2d(x, param[0], param[1], param[2])
            elif name == "avg_pool2d":
                x = F.avg_pool2d(x, param[0], param[1], param[2])
            elif name == "identity_in":
                self.buffer = x.clone()
                x = x  # Well.
            elif name == "identity_out":
                if param != []:
                    if x.shape[2] == self.buffer.shape[2]:
                        stride, padding = 1, 0
                    else:
                        stride, padding = 2, 0
                    w, b = vars[idx], vars[idx + 1]
                    # w = vars.pop(0)
                    # b = vars.pop(0)
                    self.buffer = F.conv2d(
                        self.buffer, w, b, stride=stride, padding=padding
                    )
                    idx += 2
                # else:
                #     self.buffer = F.relu(self.buffer)
                x = x + self.buffer
            else:
                breakpoint()
                raise NotImplementedError

        # make sure variable is used properly
        assert idx == len(vars)
        assert bn_idx == len(self.vars_bn)
        return x, x_flatten

    def zero_grad(self, vars=None):
        with torch.no_grad():
            if vars is None:
                for p in self.vars:
                    if p.grad is not None:
                        p.grad.zero_()
            else:
                for p in vars:
                    if p.grad is not None:
                        p.grad.zero_()

    def parameters(self):
        return self.vars
