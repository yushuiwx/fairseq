# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from collections.abc import Collection
from dataclasses import dataclass, field
from typing import List

import torch
from fairseq.dataclass import FairseqDataclass
from omegaconf import II, DictConfig
from torch.optim.optimizer import Optimizer, required

from . import FairseqOptimizer, register_optimizer


@dataclass
class FairseqLionConfig(FairseqDataclass):
    lion_beta: str = field(
        default="(0.9, 0.99)", metadata={"help": "betas for Lion optimizer"}
    )
    weight_decay: float = field(default=0.0, metadata={"help": "weight decay"})
    # TODO common vars in parent class
    lr: List[float] = II("optimization.lr")


@register_optimizer("lion", dataclass=FairseqLionConfig)
class FairseqLion(FairseqOptimizer):
    def __init__(self, cfg: DictConfig, params):
        super().__init__(cfg)
        self._optimizer = Lion(params, **self.optimizer_config)

    @property
    def optimizer_config(self):
        """
        Return a kwarg dictionary that will be used to override optimizer
        args stored in checkpoints. This allows us to load a checkpoint and
        resume training using a different set of optimizer args, e.g., with a
        different learning rate.
        """
        return {
            "lr": self.cfg.lr[0]
            if isinstance(self.cfg.lr, Collection)
            else self.cfg.lr,
            "betas": eval(self.cfg.lion_betas),
            "weight_decay": self.cfg.weight_decay,
        }

def update_fn(p, grad, exp_avg, lr, wd, beta1, beta2):
    # stepweight decay

    p.data.mul_(1 - lr * wd)

    # weight update

    update = exp_avg.clone().mul_(beta1).add(grad, alpha = 1 - beta1).sign_()
    p.add_(update, alpha = -lr)

    # decay the momentum running average coefficient
    exp_avg.mul_(beta2).add_(grad, alpha = 1 - beta2)


def exists(val):
    return val is not None


class Lion(Optimizer):
    def __init__(self, params, lr=required, betas = (0.9, 0.99), weight_decay = 0.0,):
        defaults = dict(lr=lr, lr_old=lr, betas=betas, weight_decay=weight_decay)
        super(Lion, self).__init__(params, defaults)

    @property
    def supports_memory_efficient_fp16(self):
        return True

    @property
    def supports_flat_params(self):
        return True

    def step(self, closure=None):
        """Performs a single optimization step.

        Args:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in filter(lambda p: exists(p.grad), group['params']):

                grad, lr, wd, beta1, beta2, state = p.grad, group['lr'], group['weight_decay'], *group['betas'], self.state[p]

                # init state - exponential moving average of gradient values

                if len(state) == 0:
                    state['exp_avg'] = torch.zeros_like(p)

                exp_avg = state['exp_avg']

                self.update_fn(
                    p,
                    grad,
                    exp_avg,
                    lr,
                    wd,
                    beta1,
                    beta2
                )

        return loss
