import numpy as np
from universal.algo import Algo
from universal import algos, tools
import logging
# we would like to see algos progress
logging.basicConfig(format='%(asctime)s %(message)s', level=logging.DEBUG)


class SingleOLMAX(Algo):

    PRICE_TYPE = 'ratio'
    REPLACE_MISSING = False

    def __init__(self):
        super(SingleOLMAX, self).__init__()

    def init_weights(self, columns):
        n = len(columns)
        return np.ones(n) / n

    def step(self, x, last_w, history):
        x_min = min(x)
        if x_min < 1:
            return np.where(x == x_min, 1, 0)
        return np.zeros(len(x))


class MultiOLMAX(Algo):

    PRICE_TYPE = 'raw'
    REPLACE_MISSING = False

    def __init__(self, window=5):
        super(MultiOLMAX, self).__init__(min_history=window)

        # input check
        if window < 3:
            raise ValueError('window parameter must be >=3')

        self.window = window

    def init_weights(self, columns):
        n = len(columns)
        return np.ones(n) / n

    def step(self, x, last_w, history):
        # calculate return prediction
        x_tilde = self.predict(x, history.iloc[-self.window:])
        return self.update(x_tilde)

    def predict(self, x, history):
        """ Predict returns on next day. """
        sma = history.mean()
        return sma / x

    def update(self, x):
        x_max = max(x)

        if x_max > 1:
            return np.where(x == x_max, 1, 0)

        return np.zeros(len(x))


class MultiOLMAX2(Algo):

    PRICE_TYPE = 'raw'
    REPLACE_MISSING = False

    def __init__(self, alpha = 0.3):
        super(MultiOLMAX2, self).__init__()

        self.alpha = alpha
        self.ema = 0.

    def init_weights(self, columns):
        n = len(columns)
        return np.ones(n) / n

    def step(self, x, last_w, history):
        # calculate return prediction
        x_tilde = self.predict(x)
        return self.update(x_tilde)

    def predict(self, x):
        """ Predict returns on next day. """
    #         x_tilde = self.ema / x
    #         self.ema = self.alpha * x + (1-self.alpha) * self.ema
    #         return x_tilde
        self.ema = self.alpha * x + (1-self.alpha) * self.ema
        return self.ema / x

    def update(self, x):
        x_max = max(x)

        if x_max > 1:
            return np.where(x == x_max, 1, 0)

        return np.zeros(len(x))


class MAPGRAD(Algo):

    PRICE_TYPE = 'ratio'
    REPLACE_MISSING = False

    def __init__(self, demean=True):
        super(MAPGRAD, self).__init__(min_history=1)

        self.demean = demean
        self.gradsum = 0.

    def init_weights(self, columns):
        n = len(columns)
        return np.ones(n) / n

    def step(self, x, last_w):
        # compute gradient
        if not self.demean:
            r = x.values - 1
            rp = np.dot(last_w, r)
            g = r / (1 + rp)
        elif self.demean:
            g = x / np.dot(last_w, x)
            g -= np.mean(g)

        # update gradient sum
        self.gradsum += g

        # compute learning rate
        eta = len(x) / np.linalg.norm(self.gradsum)**2

        # update weights
        w = last_w - eta * g

        return tools.simplex_proj(w)


class AdaPAMR(Algo):
    """
    Adaptive passive-aggressive mean reversion strategy for portfolio selection.
    """

    PRICE_TYPE = 'ratio'
    REPLACE_MISSING = False

    def __init__(self, demean=True, ogd_annealing=False):
        super(AdaPAMR, self).__init__(min_history=1)

        self.demean = demean
        self.gradsum = 0.
        self.ogd_annealing = ogd_annealing
        self.t = 1

    def init_weights(self, columns):
        m = len(columns)
        return np.ones(m) / m

    def step(self, x, last_b, history):
        return self.update(last_b, x)

    def update(self, b, x):
        # compute gradient
        if not self.demean:
            r = x.values - 1
            g = r
        elif self.demean:
            g = x
            g -= np.mean(g)

        # update gradient sum
        self.gradsum += g

        # update learning rate
        if not self.ogd_annealing:
            eta = len(x) / np.linalg.norm(self.gradsum)**2
        else:
            eta = 1 / np.sqrt(self.t)

        # update portfolio
        b -= eta * g

        self.t += 1

        # project it onto simplex
        return tools.simplex_proj(b)


class AdaOLMAR(Algo):
    """
    Adaptive On-Line Portfolio Selection with Moving Average Reversion
    """

    PRICE_TYPE = 'raw'
    REPLACE_MISSING = False

    def __init__(self, window=5, demean=True, ogd_annealing=False):
        """
        :param window: Lookback window.
        """

        super(AdaOLMAR, self).__init__(min_history=window)

        # input check
        if window < 3:
            raise ValueError('window parameter must be >=3')

        self.window = window
        self.demean = demean
        self.gradsum = 0.
        self.ogd_annealing = ogd_annealing
        self.t = 1

    def init_weights(self, columns):
        m = len(columns)
        return np.ones(m) / m

    def step(self, x, last_b, history):
        # calculate return prediction
        x_tilde = self.predict(x, history.iloc[-self.window:])
        b = self.update(last_b, x_tilde)
        return b

    def predict(self, x, history):
        """ Predict returns on next day. """
        sma = history.mean()
        return sma / x

    def update(self, b, x):
        # compute gradient
        if not self.demean:
            r = x.values - 1
            g = r
        elif self.demean:
            g = x
            g -= np.mean(g)

        # update gradient sum
        self.gradsum += g

        # compute learning rate
        if not self.ogd_annealing:
            eta = len(x) / np.linalg.norm(self.gradsum)**2
        else:
            eta = 1 / np.sqrt(self.t)

        # update portfolio
        b += eta * g

        self.t += 1

        # project it onto simplex
        return tools.simplex_proj(b)


class BestStock(Algo):

    PRICE_TYPE = 'raw'

    def __init__(self):
        super(BestStock, self).__init__()

    def weights(self, S):
        b = np.where(S.iloc[-1] == max(S.iloc[-1]), 1, 0)

        # weights are proportional to price times initial weights
        w = S * b

        # normalise
        w = w.div(w.sum(axis=1), axis=0)

        # shift
        w = w.shift(1)
        w.iloc[0] = 1. / S.shape[1]

        return w
