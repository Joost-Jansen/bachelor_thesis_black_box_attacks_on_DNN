import numpy as np
import torch
import torch.nn as nn
from torch import Tensor

from estimators.estimators import GradientWrapper, NESWrapper
from advertorch.utils import clamp, batch_multiply, batch_clamp, normalize_by_pnorm, clamp_by_pnorm, batch_l1_proj
from advertorch.attacks.utils import rand_init_delta
from advertorch.attacks.iterative_projected_gradient import perturb_iterative
from advertorch.attacks.blackbox.utils import _flatten
from typing import Callable, Tuple, Type
from advertorch.attacks.base import Attack
from advertorch.attacks import PGDAttack, L2PGDAttack, L1PGDAttack, LinfPGDAttack, LinfBasicIterativeAttack, \
    L2BasicIterativeAttack, LinfMomentumIterativeAttack, L2MomentumIterativeAttack
from estimators.estimators import NESWrapper, OnePointResidualWrapper, TwoPointForwardWrapper, TwoPointBackwardWrapper, \
    FDWrapper


def gradient_estimator_wrapper(attack_type: str, predicts, loss_fns, epss, nb_iters,
                               eps_iters, rand_inits, clip_mins, clip_maxs,
                               targeteds,
                               nb_sampless: int,
                               fd_etas: float,
                               estimator_type: str,
                               query_until_success: bool
                               ) -> Attack:
    AttackCls = PGDAttack
    if attack_type == 'L2-PGD':
        AttackCls = L2PGDAttack
    elif attack_type == 'L1-PGD':
        AttackCls = L1PGDAttack
    elif attack_type == 'Linf-PGD':
        AttackCls = LinfPGDAttack
    elif attack_type == 'L2-Bas-PGD':
        AttackCls = L2BasicIterativeAttack
    elif attack_type == 'Linf-Bas-PGD':
        AttackCls = LinfBasicIterativeAttack
    elif attack_type == 'L2-Mom-PGD':
        AttackCls = L2MomentumIterativeAttack
    elif attack_type == 'Linf-Mom-PGD':
        AttackCls = LinfMomentumIterativeAttack

    class EstimateAttack(AttackCls):
        """
        :param predict: forward pass function.
        :param loss_fn: loss function.
        :param eps: maximum distortion.
        :param nb_samples: number of samples to use for gradient estimation
        :param fd_eta: step-size used for Finite Difference gradient estimation
        :param nb_iter: number of iterations.
        :param eps_iter: attack step size.
        :param rand_init: (optional bool) random initialization.
        :param clip_min: mininum value per input dimension.
        :param clip_max: maximum value per input dimension.
        :param targeted: if the attack is targeted.
        """

        # def __init__(self):
        #     super(EstimateAttack, self).__init__()
        def __init__(
                self, predict, loss_fn=None, eps=0.3, nb_iter=40,
                eps_iter=0.01, rand_init=True, clip_min=0., clip_max=1.,
                targeted=False, nb_samples=100, fd_eta=1e-2, estimator="two-point-central", query_until_succes=False):

            if estimator == 'one-point-residual':
                self.estimator_type = OnePointResidualWrapper
                self.calc_samples = nb_samples
                # First iteration is null gradient thus has an extra loop
                # nb_iter += 1
            elif estimator == 'two-point-forward':
                self.estimator_type = TwoPointForwardWrapper
                self.calc_samples = nb_samples + 1
            elif estimator == 'two-point-backward':
                self.estimator_type = TwoPointBackwardWrapper
                self.calc_samples = nb_samples + 1
            elif estimator == 'two-point-central':
                self.estimator_type = NESWrapper
                self.calc_samples = nb_samples * 2
            else:
                self.estimator_type = NESWrapper
                self.calc_samples = nb_samples * 2

            super(EstimateAttack, self).__init__(
                predict=predict, loss_fn=loss_fn, eps=eps, nb_iter=nb_iter,
                eps_iter=eps_iter, rand_init=rand_init, clip_min=clip_min,
                clip_max=clip_max, targeted=targeted)


            self.nb_samples = nb_samples
            self.fd_eta = fd_eta
            self.query_until_succes = query_until_succes
            self.total_correct = 0.0
            self.last_correct = 0.0
            self.number_of_queries = 0
            self.avg_queries = 0
            # self.estimator_type = estimator_type


        def perturb(self, x, y=None):
            """
            Given examples (x, y), returns their adversarial counterparts with
            an attack length of eps.

            :param x: input tensor.
            :param y: label tensor.
                      - if None and self.targeted=False, compute y as predicted
                        labels.
                      - if self.targeted=True, then y must be the targeted labels.
            :return: tensor containing perturbed inputs.
            """
            x, y = self._verify_and_process_inputs(x, y)
            shape, flat_x = _flatten(x)
            data_shape = tuple(shape[1:])

            def f(x):
                new_shape = (x.shape[0],) + data_shape
                input = x.reshape(new_shape)
                return self.predict(input)

            # first_fx = f(x).unsqueeze(-1)
            f_nes = self.estimator_type(
                f, nb_samples=self.nb_samples, fd_eta=self.fd_eta
            )

            delta = torch.zeros_like(flat_x)
            delta = nn.Parameter(delta)
            if self.rand_init:
                rand_init_delta(
                    delta, flat_x, self.ord, self.eps, self.clip_min, self.clip_max
                )
                delta.data = clamp(
                    flat_x + delta.data, min=self.clip_min, max=self.clip_max
                ) - flat_x

            rval = self.perturb_iterative_adapted(
                flat_x, y, f_nes, nb_iter=self.nb_iter,
                eps=self.eps, eps_iter=self.eps_iter,
                loss_fn=self.loss_fn, minimize=self.targeted,
                ord=self.ord, clip_min=self.clip_min,
                clip_max=self.clip_max, delta_init=delta,
                l1_sparsity=None
            )

            return rval.data.reshape(shape)

        def perturb_iterative_adapted(self, xvar, yvar, predict, nb_iter, eps, eps_iter, loss_fn,
                              delta_init=None, minimize=False, ord=np.inf,
                              clip_min=0.0, clip_max=1.0,
                              l1_sparsity=None):
            """
            Iteratively maximize the loss over the input. It is a shared method for
            iterative attacks including IterativeGradientSign, LinfPGD, etc.

            :param xvar: input data.
            :param yvar: input labels.
            :param predict: forward pass function.
            :param nb_iter: number of iterations.
            :param eps: maximum distortion.
            :param eps_iter: attack step size.
            :param loss_fn: loss function.
            :param delta_init: (optional) tensor contains the random initialization.
            :param minimize: (optional bool) whether to minimize or maximize the loss.
            :param ord: (optional) the order of maximum distortion (inf or 2).
            :param clip_min: mininum value per input dimension.
            :param clip_max: maximum value per input dimension.
            :param l1_sparsity: sparsity value for L1 projection.
                          - if None, then perform regular L1 projection.
                          - if float value, then perform sparse L1 descent from
                            Algorithm 1 in https://arxiv.org/pdf/1904.13000v1.pdf
            :return: tensor containing the perturbed input.
            """
            if delta_init is not None:
                delta = delta_init
            else:
                delta = torch.zeros_like(xvar)

            curr_correct = 1

            delta.requires_grad_()
            for ii in range(nb_iter):
                outputs = predict(xvar + delta)
                loss = loss_fn(outputs, yvar)
                if minimize:
                    loss = -loss

                loss.backward()
                if ord == np.inf:
                    grad_sign = delta.grad.data.sign()
                    delta.data = delta.data + batch_multiply(eps_iter, grad_sign)
                    delta.data = batch_clamp(eps, delta.data)
                    delta.data = clamp(xvar.data + delta.data, clip_min, clip_max
                                       ) - xvar.data

                elif ord == 2:
                    grad = delta.grad.data
                    grad = normalize_by_pnorm(grad)
                    delta.data = delta.data + batch_multiply(eps_iter, grad)
                    delta.data = clamp(xvar.data + delta.data, clip_min, clip_max
                                       ) - xvar.data
                    if eps is not None:
                        delta.data = clamp_by_pnorm(delta.data, ord, eps)

                elif ord == 1:
                    grad = delta.grad.data
                    abs_grad = torch.abs(grad)

                    batch_size = grad.size(0)
                    view = abs_grad.view(batch_size, -1)
                    view_size = view.size(1)
                    if l1_sparsity is None:
                        vals, idx = view.topk(1)
                    else:
                        vals, idx = view.topk(
                            int(np.round((1 - l1_sparsity) * view_size)))

                    out = torch.zeros_like(view).scatter_(1, idx, vals)
                    out = out.view_as(grad)
                    grad = grad.sign() * (out > 0).float()
                    grad = normalize_by_pnorm(grad, p=1)
                    delta.data = delta.data + batch_multiply(eps_iter, grad)

                    delta.data = batch_l1_proj(delta.data.cpu(), eps)
                    if xvar.is_cuda:
                        delta.data = delta.data.cuda()
                    delta.data = clamp(xvar.data + delta.data, clip_min, clip_max
                                       ) - xvar.data
                else:
                    error = "Only ord = inf, ord = 1 and ord = 2 have been implemented"
                    raise NotImplementedError(error)
                delta.grad.data.zero_()

                # Calculates avg queries until first succesfull attack
                if query_until_success:
                    x_adv_temp = clamp(xvar + delta, clip_min, clip_max)
                    outputs = predict(x_adv_temp)
                    _, predicted = torch.max(outputs.data, 1)
                    curr_correct = (predicted != yvar).sum().float()
                    self.number_of_queries += (ii+1) * self.calc_samples * (curr_correct - self.last_correct)
                    self.last_correct = curr_correct

            self.avg_queries = self.number_of_queries / curr_correct
            x_adv = clamp(xvar + delta, clip_min, clip_max)
            return x_adv

        def get_avg_queries(self):
            return self.avg_queries


    EstimateAttack.__name__ = AttackCls.__name__ + "_WithGradient_Estimator_" + estimator_type
    EstimateAttack.__qualname__ = AttackCls.__qualname__ + "_WithGradient_Estimator_" + estimator_type
    return EstimateAttack(predict=predicts, loss_fn=loss_fns, eps=epss, nb_iter=nb_iters,
                          eps_iter=eps_iters, rand_init=rand_inits, clip_min=clip_mins, clip_max=clip_maxs,
                          targeted=targeteds, nb_samples=nb_sampless, fd_eta=fd_etas, estimator=estimator_type, query_until_succes=query_until_success)
