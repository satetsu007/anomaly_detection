import numpy as np

class SDLE:
    def __init__(self, r, beta, M, T):
        """
        """
        self.r = r
        self.beta = beta
        self.M = M
        self.t = 1
        self.count = np.zeros((T+1, M))
        self.T_t = np.zeros((T+1, M))
        self.T = T
        self.prob = np.zeros((T+1, M))
    
    def update(self, x_t):
        """
        オンライン学習
        """
        for i in range(self.M):
            delta = 0
            if i == x_t:
                delta = 1

            self.T_t[self.t, i] = (1 - self.r) * self.T_t[self.t-1, i] + delta
            self.count[self.t, i] = self.count[self.t-1, i] + delta
            if self.r == 0:
                q = (self.T_t[self.t, i] + self.beta) / (self.t + self.M * self.beta)
                # self.prob[self.t, i] = q / self.count[self.t, i]
                # self.prob[self.t, i] = q
            else:
                q = (self.T_t[self.t, i] + self.beta) / ((1 - (1 - self.r) ** self.t) / self.r + self.M * self.beta)
                # self.prob[self.t, i] = q
                # self.prob[self.t, i] = q / self.count[self.t, i]
            
            if not self.count[self.t, i] == 0:
                self.prob[self.t, i] = q / self.count[self.t, i]

        self.t += 1
    
    def train(self, x):
        """
        バッチ学習
        """
        while self.t <= self.T:
            self.update(x[self.t-1])