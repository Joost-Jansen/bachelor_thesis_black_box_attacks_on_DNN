# Copyright (c) 2018-present, Royal Bank of Canada.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
#  Added one-point residual, two-point forward and backward estimator by Joost Jansen

import torch
import numpy as np


def norm(v):
    return torch.sqrt((v ** 2).sum(-1))


class GradientWrapper(torch.nn.Module):
    """
    Define a backward pass for a blackbox function using extra queries.
    Once wrapped, the blackbox function will become compatible with any attack
    in Advertorch, so long as self.training is True.

    Disclaimer: This wrapper assumes inputs will have shape [nbatch, ndim].
    For models that operate on images, you will need to wrap the function
    inside a reshaper.  See NESAttack for an example.

    :param func: A blackbox function.
        - This function must accept, and output, torch tensors.
    """

    def __init__(self, func, first_fx=None):
        super().__init__()
        self.func = func
        self.amount_of_queries = 0

        class _Func(torch.autograd.Function):
            @staticmethod
            def forward(ctx, input):
                # grad_est does not require grad
                output = self.func(input)
                grad_est = self.estimate_grad(input)
                ctx.save_for_backward(grad_est)
                return output

            @staticmethod
            def backward(ctx, grad_output):
                # Note: this is not general! May not work for images
                # Be careful about dimensions
                grad_est, = ctx.saved_tensors
                grad_input = None

                if ctx.needs_input_grad[0]:
                    grad_input = torch.bmm(grad_output.unsqueeze(1), grad_est)
                    grad_input = grad_input.squeeze(1)
                return grad_input

        self.diff_func = _Func.apply

    def batch_query(self, x):
        """
        Reshapes the queries for efficient, parallel estimation.
        """
        n_batch, n_dim, nb_samples = x.shape
        x = x.permute(0, 2, 1).reshape(-1, n_dim)
        outputs = self.func(x)  # shape [..., n_output]
        outputs = outputs.reshape(n_batch, nb_samples, -1)

        return outputs.permute(0, 2, 1)

    def estimate_grad(self, x):
        raise NotImplementedError

    def forward(self, x):
        if not self.training:
            output = self.func(x)
        else:
            output = self.diff_func(x)

        return output

    def get_amount_of_queries(self):
        return self.amount_of_queries


class FDWrapper(GradientWrapper):
    """
    Finite-Difference Estimator.
    For every backward pass, this module makes 2 * n_dim queries per
    instance.

    :param func: A blackbox function.
        - This function must accept, and output, torch tensors.
    :param fd_eta: Step-size used for the finite-difference estimation.
    """

    def __init__(self, func, nb_samples=None, fd_eta=1e-3, first_fx=None):
        super().__init__(func)
        self.fd_eta = fd_eta

    def estimate_grad(self, x):
        id_mat = torch.diag(torch.ones_like(x[0]))  # shape [D,D]
        fxp = self.batch_query(
            x[:, :, None] + self.fd_eta * id_mat[None, :, :]
        )

        fxm = self.batch_query(
            x[:, :, None] - self.fd_eta * id_mat[None, :, :]
        )

        grad_est = (fxp - fxm) / (2.0 * self.fd_eta)
        self.amount_of_queries += 2 * x[0].size()
        return grad_est


class NESWrapper(GradientWrapper):
    """
    Natural-evolutionary strategy for gradient estimation.
    For every backward pass, this module makes 2 * nb_samples
    queries per instance.

    :param func: A blackbox function.
        - This function must accept, and output, torch tensors.
    :param nb_samples: Number of samples to use in the grad estimation.
    :param fd_eta: Step-size used for the finite-difference estimation.
    """

    def __init__(self, func, nb_samples, fd_eta=1e-3, first_fx=None):
        super().__init__(func)
        self.nb_samples = nb_samples
        self.fd_eta = fd_eta
        self.amount_of_queries = 2 * nb_samples

    def estimate_grad(self, x, prior=None):
        # x shape: [nbatch, ndim]
        ndim = np.prod(list(x.shape[1:]))

        # [nbatch, ndim, nsamples]
        exp_noise = x.new_full(tuple(x.shape) + (self.nb_samples,), 0)
        exp_noise.normal_()
        exp_noise /= (ndim ** 0.5)

        fxp = self.batch_query(
            x.unsqueeze(-1) + self.fd_eta * exp_noise
        )

        fxm = self.batch_query(
            x.unsqueeze(-1) - self.fd_eta * exp_noise
        )

        gx_s = (fxp - fxm) / (2.0 * self.fd_eta)  # [nbatch, noutput, nsamples]

        grad_est = (gx_s[:, :, None, :] * exp_noise[:, None, :, :]).sum(-1)

        return grad_est


class OnePointResidualWrapper(GradientWrapper):
    """
    Natural-evolutionary strategy for gradient estimation.
    For every backward pass, this module makes nb_samples
    queries per instance.

    :param func: A blackbox function.
        - This function must accept, and output, torch tensors.
    :param nb_samples: Number of samples to use in the grad estimation.
    :param fd_eta: Step-size used for the finite-difference estimation.
    """

    def __init__(self, func, nb_samples, fd_eta=1e-3, first_fx=None):
        super().__init__(func)
        self.nb_samples = nb_samples
        self.fd_eta = fd_eta
        self.last_fxp = first_fx
        self.amount_of_queries = nb_samples

    def estimate_grad(self, x, prior=None):
        # x shape: [nbatch, ndim]
        ndim = np.prod(list(x.shape[1:]))

        # [nbatch, ndim, nsamples]
        exp_noise = x.new_full(tuple(x.shape) + (self.nb_samples,), 0)
        exp_noise.normal_()
        exp_noise /= (ndim ** 0.5)

        fxp = self.batch_query(
            x.unsqueeze(-1) + self.fd_eta * exp_noise
        )

        if self.last_fxp is None:
            self.last_fxp = self.func(x).unsqueeze(-1)

        gx_s = (fxp - self.last_fxp) / self.fd_eta  # [nbatch, noutput, nsamples]

        grad_est = (gx_s[:, :, None, :] * exp_noise[:, None, :, :]).sum(-1)
        self.last_fxp = torch.clone(fxp)
        return grad_est


class TwoPointForwardWrapper(GradientWrapper):
    """
    two-point forward strategy for gradient estimation.
    For every backward pass, this module makes nb_samples + nb_iter
    queries per instance.

    :param func: A blackbox function.
        - This function must accept, and output, torch tensors.
    :param nb_samples: Number of samples to use in the grad estimation.
    :param fd_eta: Step-size used for the finite-difference estimation.
    """

    def __init__(self, func, nb_samples, fd_eta=1e-3, first_fx=None):
        super().__init__(func)
        self.nb_samples = nb_samples
        self.fd_eta = fd_eta
        self.curr_fx = 0
        self.amount_of_queries = nb_samples  # must add nb_iter

    def estimate_grad(self, x, prior=None):
        # x shape: [nbatch, ndim]
        ndim = np.prod(list(x.shape[1:]))
        output = self.func(x)
        self.curr_fx = output.unsqueeze(-1)
        # [nbatch, ndim, nsamples]
        exp_noise = x.new_full(tuple(x.shape) + (self.nb_samples,), 0)
        exp_noise.normal_()
        exp_noise /= (ndim ** 0.5)

        fxp = self.batch_query(
            x.unsqueeze(-1) + self.fd_eta * exp_noise
        )

        # fxm = self.batch_query(
        #     x.unsqueeze(-1) - self.fd_eta * exp_noise
        # )

        gx_s = (fxp - self.curr_fx) / self.fd_eta  # [nbatch, noutput, nsamples]

        grad_est = (gx_s[:, :, None, :] * exp_noise[:, None, :, :]).sum(-1)

        return grad_est


class TwoPointBackwardWrapper(GradientWrapper):
    """
    two-point backward strategy for gradient estimation.
    For every backward pass, this module makes nb_samples + nb_iter
    queries per instance.

    :param func: A blackbox function.
        - This function must accept, and output, torch tensors.
    :param nb_samples: Number of samples to use in the grad estimation.
    :param fd_eta: Step-size used for the finite-difference estimation.
    """

    def __init__(self, func, nb_samples, fd_eta=1e-3, first_fx=None):
        super().__init__(func)
        self.nb_samples = nb_samples
        self.fd_eta = fd_eta
        self.curr_fx = 0
        self.amount_of_queries = nb_samples  # must add nb_iter

    def estimate_grad(self, x, prior=None):
        # x shape: [nbatch, ndim]
        ndim = np.prod(list(x.shape[1:]))
        output = self.func(x)
        self.curr_fx = output.unsqueeze(-1)
        # [nbatch, ndim, nsamples]
        exp_noise = x.new_full(tuple(x.shape) + (self.nb_samples,), 0)
        exp_noise.normal_()
        exp_noise /= (ndim ** 0.5)

        # fxp = self.batch_query(
        #     x.unsqueeze(-1) + self.fd_eta * exp_noise
        # )

        fxm = self.batch_query(
            x.unsqueeze(-1) - self.fd_eta * exp_noise
        )

        gx_s = (self.curr_fx - fxm) / self.fd_eta  # [nbatch, noutput, nsamples]

        grad_est = (gx_s[:, :, None, :] * exp_noise[:, None, :, :]).sum(-1)

        return grad_est
