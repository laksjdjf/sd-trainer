#https://github.com/lucidrains/lion-pytorch/blob/6a74fdc0ba572ab5683dc0270c66c20ecbc02d09/lion_pytorch/lion_pytorch.py

from __future__ import annotations
from typing import Tuple, Callable

import torch
from torch.optim.optimizer import Optimizer

# functions

def exists(val):
    return val is not None

# class

class Lion(Optimizer):
    def __init__(
        self,
        params,
        lr: float = 1e-4,
        weight_decay: float = 0.0,
        decoupled_weight_decay: bool = False,
    ):
        assert lr > 0.

        self._init_lr = lr
        self.decoupled_wd = decoupled_weight_decay

        defaults = dict(
            lr = lr,
            weight_decay = weight_decay
        )

        super().__init__(params, defaults)

    @torch.no_grad()
    def step(
        self,
        closure: Callable | None = None
    ):

        loss = None
        if exists(closure):
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            for p in filter(lambda p: exists(p.grad), group['params']):

                grad, lr, wd, decoupled_wd, init_lr = p.grad, group['lr'], group['weight_decay'],  self.decoupled_wd, self._init_lr

                if decoupled_wd:
                    wd /= init_lr

                p.data.mul_(1. - lr * wd).add_(grad.sign_(), alpha = -lr)
                

        return loss