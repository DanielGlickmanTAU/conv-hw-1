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
        self.grad_ctx = (M, x)
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
        M, x = self.grad_ctx
        N, C = M.shape
        rows = torch.arange(N)
        G = torch.ones_like(M)

        G[M <= 0] = 0
        G[rows, y] = - G.sum(dim=1)
        grad = torch.matmul(x.T, G) / N

        # ========================

        return grad


import torchvision.transforms as tvtf
import hw1.datasets as hw1datasets
import hw1.dataloaders as hw1dataloaders
import hw1.transforms as hw1tf

tf_ds = tvtf.Compose([
    tvtf.ToTensor(),  # Convert PIL image to pytorch Tensor
    tvtf.Normalize(
        # Normalize each chanel with precomputed mean and std of the train set
        mean=(0.49139968,),
        std=(0.24703223,)),
    hw1tf.TensorView(-1),  # Reshape to 1D Tensor
    hw1tf.BiasTrick(),  # Apply the bias trick (add bias dimension to data)
])

# Define how much data to load
num_train = 10000
num_test = 1000
batch_size = 1000

# Training dataset
ds_train = hw1datasets.SubsetDataset(
    torchvision.datasets.MNIST(root='./data/mnist/', download=True, train=True, transform=tf_ds),
    num_train)

# Create training & validation sets
dl_train, dl_valid = hw1dataloaders.create_train_validation_loaders(
    ds_train, validation_ratio=0.2, batch_size=batch_size, num_workers=0
)

import helpers.dataloader_utils as dl_utils

# Test dataset & loader
ds_test = hw1datasets.SubsetDataset(
    torchvision.datasets.MNIST(root='../data/mnist/', download=False, train=False, transform=tf_ds),
    num_test)
dl_test = torch.utils.data.DataLoader(ds_test, batch_size, num_workers=0)

x0, y0 = ds_train[0]
n_features = torch.numel(x0)
n_classes = 10
dl_test = torch.utils.data.DataLoader(ds_test, batch_size, num_workers=0)
x, y = dl_utils.flatten(dl_test)
import hw1.linear_classifier as hw1linear

lin_cls = hw1linear.LinearClassifier(n_features, n_classes)
y_pred, x_scores = lin_cls.predict(x)

loss_fn = SVMHingeLoss(delta=1)

# Compute loss and gradient
loss = loss_fn(x, y, x_scores, y_pred)
grad = loss_fn.grad()

# Test the gradient with a pre-computed expected value
expected_grad = torch.load('../tests/assets/part3_expected_grad.pt')
print(expected_grad)
diff = torch.norm(grad - expected_grad)
print('diff =', diff.item())
test = unittest.TestCase()
test.assertAlmostEqual(diff, 0, delta=1e-1)

lin_cls = hw1linear.LinearClassifier(n_features, n_classes)

# Evaluate on the test set
x_test, y_test = dl_utils.flatten(dl_test)
y_test_pred, _ = lin_cls.predict(x_test)
test_acc_before = lin_cls.evaluate_accuracy(y_test, y_test_pred)

# Train the model
svm_loss_fn = SVMHingeLoss()
train_res, valid_res = lin_cls.train(dl_train, dl_valid, svm_loss_fn,
                                     learn_rate=1e-3, weight_decay=0.5,
                                     max_epochs=31)

# Re-evaluate on the test set
y_test_pred, _ = lin_cls.predict(x_test)
test_acc_after = lin_cls.evaluate_accuracy(y_test, y_test_pred)
import matplotlib.pyplot as plt

# Plot loss and accuracy
fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(10, 5))
for i, loss_acc in enumerate(('loss', 'accuracy')):
    axes[i].plot(getattr(train_res, loss_acc))
    axes[i].plot(getattr(valid_res, loss_acc))
    axes[i].set_title(loss_acc.capitalize(), fontweight='bold')
    axes[i].set_xlabel('Epoch')
    axes[i].legend(('train', 'valid'))
    axes[i].grid(which='both', axis='y')

# Check test set accuracy
print(f'Test-set accuracy before training: {test_acc_before:.1f}%')
print(f'Test-set accuracy after training: {test_acc_after:.1f}%')
test.assertGreaterEqual(test_acc_after, 80.0)
