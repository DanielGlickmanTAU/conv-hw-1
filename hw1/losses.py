import abc
import unittest

import torch
import torchvision


class ClassifierLoss(abc.ABC):
    """
    Represents a loss function of a classifier.
    """

    def __call__(self, *args, **kwargs):
        return self.loss(*args, **kwargs)

    @abc.abstractmethod
    def loss(self, *args, **kw):
        pass

    @abc.abstractmethod
    def grad(self):
        """
        :return: Gradient of the last calculated loss w.r.t. model
            parameters, as a Tensor of shape (D, C).
        """
        pass


class SVMHingeLoss(ClassifierLoss):
    def __init__(self, delta=1.0):
        self.delta = delta
        self.grad_ctx = {}

    def loss(self, x, y, x_scores, y_predicted):
        """
        Calculates the loss for a batch of samples.

        :param x: Batch of samples in a Tensor of shape (N, D).
        :param y: Ground-truth labels for these samples: (N,)
        :param x_scores: The predicted class score for each sample: (N, C).
        :param y_predicted: The predicted class label for each sample: (N,).
        :return: The classification loss as a Tensor of shape (1,).
        """

        assert x_scores.shape[0] == y.shape[0]
        assert y.dim() == 1

        # TODO: Implement SVM loss calculation based on the hinge-loss formula.
        #
        # Notes:
        # - Use only basic pytorch tensor operations, no external code.
        # - Partial credit will be given for an implementation with only one
        #   explicit loop.
        # - Full credit will be given for a fully vectorized implementation
        #   (zero explicit loops).
        #   Hint: Create a matrix M where M[i,j] is the margin-loss
        #   for sample i and class j (i.e. s_j - s_{y_i} + delta).

        loss = None
        # ====== YOUR CODE: ======
        N = y.shape[0]
        row_index = torch.arange(N)
        s_y_i = x_scores[row_index, y]
        s_y_i_as_matrix = s_y_i.unsqueeze(1)
        M = self.delta + x_scores - s_y_i_as_matrix
        M[row_index, y] = 0
        hinge_loss = torch.max(M, torch.zeros_like(M)).sum(dim=1)
        loss = hinge_loss.mean()
        # ========================

        # TODO: Save what you need for gradient calculation in self.grad_ctx
        self.grad_ctx = (M, x, y)
        # raise NotImplementedError()
        # ========================

        return loss

    def grad(self):
        # TODO: Implement SVM loss gradient calculation
        # Same notes as above. Hint: Use the matrix M from above, based on
        # it create a matrix G such that X^T * G is the gradient.

        grad = None
        # ====== YOUR CODE: ======
        # x shape is (N,D)
        # M shape (N,C)
        # y shape N
        M, x, y = self.grad_ctx
        N, C = M.shape
        rows_index = torch.arange(N)

        G = torch.ones_like(M)
        G[M <= 0] = 0
        G[rows_index, y] = - G.sum(dim=1)
        grad = torch.matmul(x.T, G) / N

        # ========================

        return grad