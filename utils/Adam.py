import numpy as np

class Adam:
    def __init__(self, learning_rate=1e-4, beta1=0.9, beta2=0.999, eps=1e-8):
        self.lr, self.beta1, self.beta2, self.eps = learning_rate, beta1, beta2, eps
        self.t = 0
        self.m = None
        self.v = None

    def step(self, params, grad):
        if self.m is None:
            self.m = np.zeros_like(params, dtype=float)  # m₀: one zero per parameter
            self.v = np.zeros_like(params, dtype=float)  # v₀: one zero per parameter

        self.t += 1
        self.m = self.beta1 * self.m + (1 - self.beta1) * grad
        self.v = self.beta2 * self.v + (1 - self.beta2) * grad**2

        m_hat = self.m / (1 - self.beta1**self.t)
        v_hat = self.v / (1 - self.beta2**self.t)

        return params - self.lr * m_hat / (np.sqrt(v_hat) + self.eps)
