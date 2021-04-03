from builtins import range
import numpy as np
from random import shuffle
from past.builtins import xrange

def softmax_loss_naive(W, X, y, reg):
    """
    Softmax loss function, naive implementation (with loops)

    Inputs have dimension D, there are C classes, and we operate on minibatches
    of N examples.

    Inputs:
    - W: A numpy array of shape (D, C) containing weights.
    - X: A numpy array of shape (N, D) containing a minibatch of data.
    - y: A numpy array of shape (N,) containing training labels; y[i] = c means
      that X[i] has label c, where 0 <= c < C.
    - reg: (float) regularization strength

    Returns a tuple of:
    - loss as single float
    - gradient with respect to weights W; an array of same shape as W
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)

    #############################################################################
    # TODO: Compute the softmax loss and its gradient using explicit loops.     #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    num_train = X.shape[0]
    num_class = W.shape[1]
    for i in range(num_train):
      prod = X[i].dot(W)
      prod -= max(prod)
      exp_scores = np.exp(prod)
      den = np.sum(exp_scores)
      num = exp_scores[y[i]]
      dloss = np.log(num/den)
      loss -= dloss

      # calcualte for dW 
      d = -den/num
      for j in range(num_class): 
        dW[:,j] -= d * (exp_scores[y[i]] * exp_scores[j] / (den ** 2)) * X[i].T
        if y[i] == j: 
          dW[:, j] += d * exp_scores[j] / den * X[i].T

    loss /= num_train
    loss += reg * np.sum(W*W)

    dW /= num_train 
    dW += (2 * reg * W)

    pass

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
    """
    Softmax loss function, vectorized version.

    Inputs and outputs are the same as softmax_loss_naive.
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)

    #############################################################################
    # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    num_train = X.shape[0]
    prod = X.dot(W)
    prod -= np.max(prod, axis = 1).reshape(num_train, -1)
    exp_prod = np.exp(prod)
    exp_prod_sum = np.sum(exp_prod, axis = 1)
    exp_prod_y = exp_prod[np.arange(num_train), y]
    exp_loss = exp_prod_y/exp_prod_sum
    loss = - np.sum(np.log(exp_loss))
    loss /= num_train 
    loss += reg * np.sum(W*W)

    # calculate dW 
    exp_loss_inverse = (1/exp_loss).reshape(num_train, 1)
    sum_inverse = (1/exp_prod_sum).reshape(num_train, 1)
    exp_prod_y = exp_prod_y.reshape(num_train, 1)
    dW = X.T.dot(exp_prod_y * exp_prod * (sum_inverse * sum_inverse) * exp_loss_inverse) 
    mask = np.zeros_like(prod)
    mask[np.arange(num_train), y] = 1 
    dW -= X.T.dot(exp_prod_y * mask * sum_inverse * exp_loss_inverse)
    dW /= num_train 
    dW += (2 * reg * W)


    pass

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW
