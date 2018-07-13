import numpy as np
import scipy.stats as st

class SDEM:
    def __init__(self, r, alpha, k, T, d):
        """
        初期化
        """
        self.r = r
        self.alpha = alpha
        self.k = k
        self.T = T+1
        self.d = d
        
        self.prob = np.zeros((self.T, self.k))
        self.mu = np.zeros((self.T, self.k, self.d))
        self.mu_ = np.zeros((self.T, self.k, self.d))
        self.pi = np.zeros((self.T, self.k))
        self.sigma = np.zeros((self.T, self.k, self.d, self.d))
        self.sigma_ = np.zeros((self.T, self.k, self.d, self.d))
        
        for i in range(self.k):
            self.pi[0, i] = 1 / self.k
            self.mu[0, i] = np.random.uniform(size=self.d)
            self.mu_[0, i] = self.mu[0, i] * self.pi[0, i]
            mu_tmp = self.mu[0, i].reshape((self.d, 1))
            self.sigma[0, i] = np.identity(self.d)# * 0.1
            self.sigma_[0, i] = (self.sigma[0, i] + np.dot(mu_tmp, mu_tmp.T)) * self.pi[0, i]
        self.t = 1
    
    def calc_prob(self, idy, pi, mu, sigma):
        p_ = np.zeros(self.k)
        p = np.zeros(self.k)
        for i in range(self.k):
            p[i] = st.multivariate_normal.pdf(idy, mu[i], sigma[i])
            p_[i] = pi[i]*p[i]
        return p, p_
    
    def update(self, idy):
        """
        更新
        """
        t = self.t
        for i in range(self.k):
            _, p_ = self.calc_prob(idy, self.pi[t-1], self.mu[t-1], self.sigma[t-1])
            gamma = ((1 - self.alpha * self.r) * (p_[i] / np.sum(p_))) + ((self.alpha * self.r) / self.k)
            
            self.pi[t, i] = ((1 - self.r) * self.pi[t-1, i]) + (self.r * gamma)
            self.mu_[t, i] = ((1 - self.r) * self.mu_[t-1, i]) + ((self.r * gamma) * idy)
            self.mu[t, i] = self.mu_[t, i] / self.pi[t, i]
            
            idy_tmp = idy.reshape((self.d, 1))
            mu_tmp = self.mu[t, i].reshape((self.d, 1))
            
            self.sigma_[t, i] = ((1 - self.r) * self.sigma_[t-1, i]) + ((self.r * gamma) * np.dot(idy_tmp, idy_tmp.T))
            self.sigma[t, i] = (self.sigma_[t, i] / self.pi[t, i]) - np.dot(mu_tmp, mu_tmp.T)
        p, _ = self.calc_prob(idy, self.pi[t], self.mu[t], self.sigma[t])
        self.prob[t] = p
            
        self.t += 1
        
    def skip(self):
        """
        更新しない(前時点のパラメータを引き継ぐ)
        """
        t = self.t
        for i in range(self.k):
            self.pi[t, i] = self.pi[t-1, i]
            self.mu[t, i] = self.mu[t-1, i]
            self.mu_[t, i] = self.mu_[t-1, i]
            self.sigma[t, i] = self.sigma[t-1, i]
            self.sigma_[t, i] = self.sigma_[t-1, i]
            self.prob[t] = self.prob[t-i, i]
        self.t += 1
    
    def train(self, y):
        """
        バッチ学習
        """
        while self.t < self.T:
            self.update(y[self.t-1])