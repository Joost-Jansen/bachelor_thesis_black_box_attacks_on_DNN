import numpy as np
import torch
import torch.nn as nn
from torch import Tensor

from estimators.estimators import GradientWrapper, NESWrapper
from advertorch.utils import clamp, normalize_by_pnorm, batch_multiply, batch_clamp
from advertorch.attacks.utils import rand_init_delta
from advertorch.attacks.iterative_projected_gradient import perturb_iterative
from advertorch.attacks.blackbox.utils import _flatten
from typing import Callable, Tuple, Type
from advertorch.attacks.base import Attack
from advertorch.attacks import PGDAttack, L2PGDAttack, L1PGDAttack, LinfPGDAttack, LinfBasicIterativeAttack, \
    L2BasicIterativeAttack, LinfMomentumIterativeAttack, L2MomentumIterativeAttack
from estimators.estimators import NESWrapper, OnePointResidualWrapper, TwoPointForwardWrapper, TwoPointBackwardWrapper, \
    FDWrapper


def gradient_estimator_wrapper_mia(attack_type: str, predicts, loss_fns, epss, nb_iters,
                                   eps_iters, clip_mins, clip_maxs,
                                   targeteds,
                                   nb_sampless: int,
                                   fd_etas: float,
                                   estimator_type: str,
                                   query_until_success: bool
                                   ) -> Attack:
    AttackCls = PGDAttack

    if attack_type == 'L2-MOM':
        AttackCls = L2MomentumIterativeAttack
    elif attack_type == 'Linf-MOM':
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
                eps_iter=0.01, clip_min=0., clip_max=1.,
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
                eps_iter=eps_iter, clip_min=clip_min,
                clip_max=clip_max, targeted=targeted)

            self.predict = predict
            self.nb_samples = nb_samples
            self.fd_eta = fd_eta
            self.query_until_succes = query_until_succes
            self.avg_queries = 0.
            self.last_correct = 0.
            self.number_of_queries = 0
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

            self.loss_fn = f_nes

            delta = torch.zeros_like(flat_x)
            g = torch.zeros_like(flat_x)

            delta = nn.Parameter(delta)

            curr_correct = 0.

            for i in range(self.nb_iter):

                if delta.grad is not None:
                    delta.grad.detach_()
                    delta.grad.zero_()

                imgadv = x + delta
                outputs = self.predict(imgadv)
                loss = self.loss_fn(outputs, y)
                if self.targeted:
                    loss = -loss
                loss.backward()

                g = self.decay_factor * g + normalize_by_pnorm(
                    delta.grad.data, p=1)
                # according to the paper it should be .sum(), but in their
                #   implementations (both cleverhans and the link from the paper)
                #   it is .mean(), but actually it shouldn't matter
                if self.ord == np.inf:
                    delta.data += batch_multiply(self.eps_iter, torch.sign(g))
                    delta.data = batch_clamp(self.eps, delta.data)
                    delta.data = clamp(
                        x + delta.data, min=self.clip_min, max=self.clip_max) - x
                elif self.ord == 2:
                    delta.data += self.eps_iter * normalize_by_pnorm(g, p=2)
                    delta.data *= clamp(
                        (self.eps * normalize_by_pnorm(delta.data, p=2) /
                         delta.data),
                        max=1.)
                    delta.data = clamp(
                        x + delta.data, min=self.clip_min, max=self.clip_max) - x
                else:
                    error = "Only ord = inf and ord = 2 have been implemented"
                    raise NotImplementedError(error)
                    # Calculates avg queries until first succesfull attack

                if query_until_success:
                    x_adv_temp = clamp(x + delta.data, min=self.clip_min, max=self.clip_max)
                    outputs = self.predict(x_adv_temp)
                    _, predicted = torch.max(outputs.data, 1)
                    curr_correct = (predicted != y).sum().float()
                    self.number_of_queries += (i + 1) * self.calc_samples * (curr_correct - self.last_correct)
                    self.last_correct = curr_correct

                self.avg_queries = self.number_of_queries / curr_correct

            rval = x + delta.data
            return rval.data.reshape(shape)

    EstimateAttack.__name__ = AttackCls.__name__ + "_WithGradient_Estimator_" + estimator_type
    EstimateAttack.__qualname__ = AttackCls.__qualname__ + "_WithGradient_Estimator_" + estimator_type
    return EstimateAttack(predict=predicts, loss_fn=loss_fns, eps=epss, nb_iter=nb_iters,
                          eps_iter=eps_iters, clip_min=clip_mins, clip_max=clip_maxs,
                          targeted=targeteds, nb_samples=nb_sampless, fd_eta=fd_etas, estimator=estimator_type,
                          query_until_succes=query_until_success)
