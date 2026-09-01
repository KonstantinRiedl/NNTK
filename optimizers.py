import time

import torch
from torch.optim import Optimizer

from linear_algebra import (
    conjugate_gradient,
    diagonal_GN_preconditioner,
    diagonal_Hessian_preconditioner,
    gnvp_explicit,
    hvp_explicit,
)
from model import flatten, unflatten


def _trainable_flatten(model, tensors):
    return flatten([t for t, p in zip(tensors, model.parameters()) if p.requires_grad])


class GradientDescentOptimizer(Optimizer):
    def __init__(self, params, lr=1.0):
        super().__init__(params, dict(lr=lr))
        self.timers = {"grad_total": 0.0, "overhead": 0.0}

    def step(self, closure, x, y, model):
        t0 = time.perf_counter()
        with torch.enable_grad():
            loss = closure()

        grad = flatten(
            torch.autograd.grad(loss, self.param_groups[0]["params"], create_graph=False)
        ).detach()

        self.timers["overhead"] += time.perf_counter() - t0
        return loss, grad

    @torch.no_grad()
    def apply_step(self, step):
        lr = self.param_groups[0]["lr"]
        idx = 0
        for p in self.param_groups[0]["params"]:
            n = p.numel()
            p.add_(step[idx:idx + n].view_as(p), alpha=-lr)
            idx += n


class RMSpropOptimizer(Optimizer):
    def __init__(self, params, lr=1e-3, alpha=0.99, eps=1e-8):
        super().__init__(params, dict(lr=lr, alpha=alpha, eps=eps))
        self.v = None
        self.timers = {"grad_total": 0.0, "overhead": 0.0}

    def step(self, closure, x, y, model):
        t0 = time.perf_counter()
        with torch.enable_grad():
            loss = closure()

        grad = flatten(
            torch.autograd.grad(loss, self.param_groups[0]["params"], create_graph=False)
        ).detach()

        if self.v is None:
            self.v = torch.zeros_like(grad)

        alpha = self.param_groups[0]["alpha"]
        eps = self.param_groups[0]["eps"]
        self.v.mul_(alpha).addcmul_(grad, grad, value=1 - alpha)
        step = grad / (torch.sqrt(self.v) + eps)

        self.timers["overhead"] += time.perf_counter() - t0
        return loss, step

    @torch.no_grad()
    def apply_step(self, step):
        lr = self.param_groups[0]["lr"]
        idx = 0
        for p in self.param_groups[0]["params"]:
            n = p.numel()
            p.add_(step[idx:idx + n].view_as(p), alpha=-lr)
            idx += n


class AdamOptimizer(Optimizer):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8):
        super().__init__(params, dict(lr=lr, betas=betas, eps=eps))
        self.t = 0
        self.m = None
        self.v = None
        self.timers = {"grad_total": 0.0, "overhead": 0.0}

    def step(self, closure, x, y, model):
        t0 = time.perf_counter()
        with torch.enable_grad():
            loss = closure()

        grad = flatten(
            torch.autograd.grad(loss, self.param_groups[0]["params"], create_graph=False)
        ).detach()

        if self.m is None:
            self.m = torch.zeros_like(grad)
            self.v = torch.zeros_like(grad)

        beta1, beta2 = self.param_groups[0]["betas"]
        eps = self.param_groups[0]["eps"]

        self.t += 1
        self.m.mul_(beta1).add_(grad, alpha=1 - beta1)
        self.v.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

        m_hat = self.m / (1 - beta1 ** self.t)
        v_hat = self.v / (1 - beta2 ** self.t)
        step = m_hat / (torch.sqrt(v_hat) + eps)

        self.timers["overhead"] += time.perf_counter() - t0
        return loss, step

    @torch.no_grad()
    def apply_step(self, step):
        lr = self.param_groups[0]["lr"]
        idx = 0
        for p in self.param_groups[0]["params"]:
            n = p.numel()
            p.add_(step[idx:idx + n].view_as(p), alpha=-lr)
            idx += n


class GaussNewtonOptimizer(Optimizer):
    def __init__(
        self,
        params,
        damping=1e-3,
        lr=1.0,
        preconditioned=False,
        max_retries=6,
        damping_decrease=0.8,
        damping_increase=2.0,
        damping_min=1e-12,
        damping_max=1e2,
    ):
        super().__init__(params, dict(damping=damping, lr=lr))
        self.cg_history = []
        self.timers = {"gnvp_total": 0.0, "cg": 0.0, "overhead": 0.0}
        self.preconditioned = preconditioned
        self.max_retries = max_retries
        self.damping_decrease = damping_decrease
        self.damping_increase = damping_increase
        self.damping_min = damping_min
        self.damping_max = damping_max

    def step(self, closure, x, y, model):
        t0 = time.perf_counter()
        with torch.enable_grad():
            loss = closure()

        grad = flatten(
            torch.autograd.grad(loss, self.param_groups[0]["params"], create_graph=False)
        ).detach()
        damping = self.param_groups[0]["damping"]
        self.timers["overhead"] += time.perf_counter() - t0

        def GNVP_solver(v):
            t = time.perf_counter()
            out = _trainable_flatten(model, gnvp_explicit(model, x, unflatten(v, model)))
            self.timers["gnvp_total"] += time.perf_counter() - t
            return out

        ordered_params = list(self.param_groups[0]["params"])
        t0 = time.perf_counter()
        iters = 0

        for _ in range(self.max_retries):
            preconditioner = (
                diagonal_GN_preconditioner(model, x, damping=damping)
                if self.preconditioned
                else None
            )

            step_dir, iters, status = conjugate_gradient(
                GNVP_solver,
                grad,
                damping,
                preconditioner=preconditioner,
            )

            if status == "negative_curvature" or step_dir is None:
                damping = min(damping * self.damping_increase, self.damping_max)
                continue

            old_params = [p.data.clone() for p in ordered_params]
            self.apply_step(step_dir)

            with torch.no_grad():
                loss_new = torch.sum((model(x) - y) ** 2) / (2 * y.size(0))
                loss_new = loss_new.item()

            if loss_new < loss.item():
                damping = max(damping * self.damping_decrease, self.damping_min)
                self.param_groups[0]["damping"] = damping
                self.timers["cg"] += time.perf_counter() - t0
                self.cg_history.append(iters)
                return loss, step_dir

            for p, old_p in zip(ordered_params, old_params):
                p.data.copy_(old_p)

            damping = min(damping * self.damping_increase, self.damping_max)

        self.param_groups[0]["damping"] = damping
        self.timers["cg"] += time.perf_counter() - t0
        self.cg_history.append(iters)
        return loss, torch.zeros_like(grad)

    @torch.no_grad()
    def apply_step(self, step):
        lr = self.param_groups[0]["lr"]
        idx = 0
        for p in self.param_groups[0]["params"]:
            n = p.numel()
            p.add_(step[idx:idx + n].view_as(p), alpha=-lr)
            idx += n


class NewtonOptimizer(Optimizer):
    def __init__(
        self,
        params,
        damping=1e-3,
        lr=1.0,
        preconditioned=False,
        max_retries=6,
        damping_decrease=0.8,
        damping_increase=2.0,
        damping_min=1e-12,
        damping_max=1e2,
    ):
        super().__init__(params, dict(damping=damping, lr=lr))
        self.cg_history = []
        self.timers = {"hvp_total": 0.0, "cg": 0.0, "overhead": 0.0}
        self.preconditioned = preconditioned
        self.max_retries = max_retries
        self.damping_decrease = damping_decrease
        self.damping_increase = damping_increase
        self.damping_min = damping_min
        self.damping_max = damping_max

    def step(self, closure, x, y, model):
        t0 = time.perf_counter()
        with torch.enable_grad():
            loss = closure()

        grad = flatten(
            torch.autograd.grad(loss, self.param_groups[0]["params"], create_graph=False)
        ).detach()
        damping = self.param_groups[0]["damping"]
        self.timers["overhead"] += time.perf_counter() - t0

        def HVP_solver(v):
            t = time.perf_counter()
            out = _trainable_flatten(model, hvp_explicit(model, x, y, unflatten(v, model)))
            self.timers["hvp_total"] += time.perf_counter() - t
            return out

        ordered_params = list(self.param_groups[0]["params"])
        t0 = time.perf_counter()
        iters = 0

        for _ in range(self.max_retries):
            preconditioner = (
                diagonal_Hessian_preconditioner(model, x, y, damping=damping)
                if self.preconditioned
                else None
            )

            step_dir, iters, status = conjugate_gradient(
                HVP_solver,
                grad,
                damping,
                preconditioner=preconditioner,
            )

            if status == "negative_curvature" or step_dir is None:
                damping = min(damping * self.damping_increase, self.damping_max)
                continue

            old_params = [p.data.clone() for p in ordered_params]
            self.apply_step(step_dir)

            with torch.no_grad():
                loss_new = torch.sum((model(x) - y) ** 2) / (2 * y.size(0))
                loss_new = loss_new.item()

            if loss_new < loss.item():
                damping = max(damping * self.damping_decrease, self.damping_min)
                self.param_groups[0]["damping"] = damping
                self.timers["cg"] += time.perf_counter() - t0
                self.cg_history.append(iters)
                return loss, step_dir

            for p, old_p in zip(ordered_params, old_params):
                p.data.copy_(old_p)

            damping = min(damping * self.damping_increase, self.damping_max)

        self.param_groups[0]["damping"] = damping
        self.timers["cg"] += time.perf_counter() - t0
        self.cg_history.append(iters)
        return loss, torch.zeros_like(grad)

    @torch.no_grad()
    def apply_step(self, step):
        lr = self.param_groups[0]["lr"]
        idx = 0
        for p in self.param_groups[0]["params"]:
            n = p.numel()
            p.add_(step[idx:idx + n].view_as(p), alpha=-lr)
            idx += n


def make_optimizer(name, model, cfg):
    params = [p for p in model.parameters() if p.requires_grad]

    if name == "GD":
        return GradientDescentOptimizer(params, lr=cfg.gd_lr)
    if name == "Adam":
        return AdamOptimizer(params, lr=cfg.adam_lr)
    if name == "Newton":
        return NewtonOptimizer(params, damping=cfg.damping0, lr=1.0, preconditioned=False)
    if name == "GN":
        return GaussNewtonOptimizer(params, damping=cfg.damping0, lr=1.0, preconditioned=False)

    raise ValueError(f"Unknown optimizer: {name}")
