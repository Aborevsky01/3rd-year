from dataclasses import dataclass
from enum import auto
from enum import Enum
from typing import Dict
from typing import Type

import numpy as np


@dataclass
class LearningRate:
    lambda_: float = 1e-3
    s0: float = 1
    p: float = 0.5

    iteration: int = 0

    def __call__(self):
        self.iteration += 1
        return self.lambda_ * (self.s0 / (self.s0 + self.iteration)) ** self.p


class LossFunction(Enum):
    MSE = auto()
    MAE = auto()
    LogCosh = auto()
    Huber = auto()


class BaseDescent:

    def __init__(self, dimension: int, lambda_: float = 1e-3, loss_function: LossFunction = LossFunction.MSE):
        self.w: np.ndarray = np.random.rand(dimension)
        self.lr: LearningRate = LearningRate(lambda_=lambda_)
        self.loss_function: LossFunction = loss_function

    def step(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        return self.update_weights(self.calc_gradient(x, y))

    def update_weights(self, gradient: np.ndarray) -> np.ndarray:
        pass

    def calc_gradient(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        pass

    def calc_loss(self, x: np.ndarray, y: np.ndarray) -> float:
        if self.loss_function == LossFunction.MSE:
            sqr = y - x @ self.w
            return (1 / y.shape[0]) * (sqr.T @ sqr)
        if self.loss_function == LossFunction.LogCosh:
            return (1 / y.shape[0]) *  np.sum(np.log(np.cosh(y - x @ self.w)))

    def predict(self, x: np.ndarray) -> np.ndarray:
        return x.dot(self.w)
    

class VanillaGradientDescent(BaseDescent):

    def update_weights(self, gradient: np.ndarray) -> np.ndarray:
        delta = self.lr() * gradient
        self.w -= delta
        return -delta
    
    def calc_gradient(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        if self.loss_function == LossFunction.MSE:
            tmp = y - x @ self.w
            return (-2/y.shape[0]) * (tmp.T @ x)
        if self.loss_function == LossFunction.LogCosh:
            return (-1/y.shape[0]) * (x.T @ np.tanh(y - x @ self.w))

class StochasticDescent(VanillaGradientDescent):
    def __init__(self, dimension: int, lambda_: float = 1e-3, batch_size: int = 50, loss_function: LossFunction = LossFunction.MSE):
        super().__init__(dimension, lambda_, loss_function)
        self.batch_size = batch_size

    def calc_gradient(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        random_ind = np.random.randint(x.shape[0], size=self.batch_size)
        upd_x = x[random_ind]
        upd_y = y[random_ind]
        return super().calc_gradient(upd_x, upd_y)

class MomentumDescent(VanillaGradientDescent):

    def __init__(self, dimension: int, lambda_: float = 1e-3, loss_function: LossFunction = LossFunction.MSE):
        super().__init__(dimension, lambda_, loss_function)
        self.alpha: float = 0.9

        self.h: np.ndarray = np.zeros(dimension)

    def update_weights(self, gradient: np.ndarray) -> np.ndarray:
        self.h = self.h * self.alpha + self.lr() * gradient
        self.w -= self.h
        return -self.h


class Adam(VanillaGradientDescent):
    """
    Adaptive Moment Estimation gradient descent class
    """

    def __init__(self, dimension: int, lambda_: float = 1e-3, loss_function: LossFunction = LossFunction.MSE):
        super().__init__(dimension, lambda_, loss_function)
        self.eps: float = 1e-8

        self.m: np.ndarray = np.zeros(dimension)
        self.v: np.ndarray = np.zeros(dimension)

        self.beta_1: float = 0.9
        self.beta_2: float = 0.999

        self.iteration: int = 0

    def update_weights(self, gradient: np.ndarray) -> np.ndarray:
        self.m = self.beta_1 * self.m + (1-self.beta_1) * gradient
        self.v = self.beta_2 * self.v + (1-self.beta_2) * (gradient) ** 2
        self.iteration += 1
        adam_moment = self.lr() * (self.m / (1-self.beta_1 ** self.iteration)) / (np.sqrt(self.v/(1-self.beta_2 ** self.iteration)) + self.eps)
        self.w -= adam_moment
        return -adam_moment
    

class BaseDescentReg(BaseDescent):
    """
    A base class with regularization
    """

    def __init__(self, dimension: int, lambda_: float = 1e-3, mu: float = 0):
        """
        :param mu: regularization coefficient (float)
        """
        super().__init__(dimension, lambda_)

        self.mu = mu

    def calc_gradient(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        Calculate gradient of loss function and L2 regularization with respect to weights
        """
        l2_gradient: np.ndarray = self.w
        l2_gradient[-1] = 0

        return super().calc_gradient(x, y) + l2_gradient * self.mu


class VanillaGradientDescentReg(BaseDescentReg, VanillaGradientDescent):
    """
    Full gradient descent with regularization class
    """


class StochasticDescentReg(BaseDescentReg, StochasticDescent):
    """
    Stochastic gradient descent with regularization class
    """


class MomentumDescentReg(BaseDescentReg, MomentumDescent):
    """
    Momentum gradient descent with regularization class
    """


class AdamReg(BaseDescentReg, Adam):
    """
    Adaptive gradient algorithm with regularization class
    """


def get_descent(descent_config: dict) -> BaseDescent:
    descent_name = descent_config.get('descent_name', 'full')
    regularized = descent_config.get('regularized', False)

    descent_mapping: Dict[str, Type[BaseDescent]] = {
        'full': VanillaGradientDescent if not regularized else VanillaGradientDescentReg,
        'stochastic': StochasticDescent if not regularized else StochasticDescentReg,
        'momentum': MomentumDescent if not regularized else MomentumDescentReg,
        'adam': Adam if not regularized else AdamReg
    }

    if descent_name not in descent_mapping:
        raise ValueError(f'Incorrect descent name, use one of these: {descent_mapping.keys()}')

    descent_class = descent_mapping[descent_name]

    return descent_class(**descent_config.get('kwargs', {}))
