import numpy as np
import numpy.random as rd
import scipy.stats as st

class SDEM:
    def __init__(self, r, alpha, k):
        """
        パラメータの初期化
        """
        self.r = r
        self.alpha = alpha
        self.k = k
        self.d = None

        self.prob = None
        self.mu = None
        self.mu_ = None
        self.pi = None
        self.sigma = None
        self.sigma_ = None

        self.t = 1

    def calc_prob(self, y_t, mu, sigma):
        """
        t時点のパラメータを使用し確率値の計算
        """
        p = np.zeros(self.k)
        for i in range(self.k):
            p[i] = st.multivariate_normal.pdf(y_t, mu[i], sigma[i])
        return p

    #E-Step
    def E_step(self, y_t, t):
        """Eステップ(負担率gammaから各パラメータ_とpiを求める)"""

        #pi*p(y|mu,sigma)を計算する
        pi_prob = np.array([self.pi[t-1, i]*st.multivariate_normal.pdf(y_t, self.mu[t-1,i], self.sigma[t-1, i]) for i in range(self.k)])
        gamma = (1 - self.alpha * self.r) * pi_prob / np.sum(pi_prob) + (self.alpha * self.r / self.k)
        #piを計算する
        self.pi[t] = (1 - self.r) * self.pi[self.t-1] + self.r * gamma
        #mu_を計算する
        self.mu_[t] = (1 - self.r)*self.mu_[t-1] + self.r * gamma[:, np.newaxis] * y_t
        #sigma_を計算する
        for i in range(self.k):
            self.sigma_[t, i] = (1 - self.r) * self.sigma_[t-1, i] + self.r * gamma[i] * np.dot(y_t[:, np.newaxis], y_t[:, np.newaxis].T)

    #M-Step
    def M_step(self, y_t, t):
        """Mステップ(gammaを使って、各パラメータを更新する)"""
        #muを計算する
        self.mu[t] = self.mu_[t] / self.pi[t][:, np.newaxis]
        #sigmaを計算する
        for i in range(self.k):
            self.sigma[t, i] = self.sigma_[t, i] / self.pi[t, i] - np.dot(self.mu[t, i][:, np.newaxis], self.mu[t, i][:, np.newaxis].T)

    def update(self, y_t):
        """
        """
        if self.t==1:
            self.set_initial_params(y_t.shape[0])
        
        self.add_new_index()

        #E-step
        self.E_step(y_t, self.t)
        #M-Step
        self.M_step(y_t, self.t)

        self.prob[self.t] = self.calc_prob(y_t, self.mu[self.t], self.sigma[self.t])

        self.t += 1
    
    def add_new_index(self):
        """pi, mu, mu_, sigma, sigma_, probに1行追加"""
                
        prob_new = np.zeros((1, self.k))
        pi_new = np.zeros((1, self.k))
        mu_new = np.zeros((1, self.k, self.d))
        sigma_new = np.zeros((1, self.k, self.d, self.d))

        self.prob = np.concatenate([self.prob, prob_new])
        self.pi = np.concatenate([self.pi, pi_new])
        self.mu = np.concatenate([self.mu, mu_new])
        self.mu_ = np.concatenate([self.mu_, mu_new])
        self.sigma = np.concatenate([self.sigma, sigma_new])
        self.sigma_ = np.concatenate([self.sigma_, sigma_new])
    
    def set_initial_params(self, d):
        """"""
        self.d = d
        self.prob = np.zeros((1, self.k))
        self.mu = np.zeros((1, self.k, d))
        self.mu_ = np.zeros((1, self.k, d))
        self.pi = np.zeros((1, self.k))
        self.sigma = np.zeros((1, self.k, d, d))
        self.sigma_ = np.zeros((1, self.k, d, d))

        for i in range(self.k):
            self.pi[0, i] = 1 / self.k #piの初期化
            self.mu[0, i] = rd.uniform(low=0, high=1, size=d) #muの初期化(一様分布)
            self.mu_[0, i] = self.mu[0, i] * self.pi[0, i] #mu_の初期化(mu*piで計算)
            self.sigma[0, i] = np.identity(d) #sigmaの初期化(単位行列)
            self.sigma_[0, i] = (self.sigma[0, i] + np.dot(self.mu[0, i][:,np.newaxis], self.mu[0, i][:,np.newaxis].T)) * self.pi[0, i] #sigma_初期化
        
    def skip(self, y_t):
        """
        """
        if self.t==1:
            self.set_initial_params(y_t.shape[0])
        
        self.add_new_index()

        # 更新をしない
        for i in range(self.k):
            self.pi[self.t, i] = self.pi[self.t-1, i]
            self.mu[self.t, i] = self.mu[self.t-1, i]
            self.mu_[self.t, i] = self.mu_[self.t-1, i]
            self.sigma[self.t, i] = self.sigma[self.t-1, i]
            self.sigma_[self.t, i] = self.sigma_[self.t-1, i]
            self.prob[self.t] = self.prob[self.t-i, i]
        self.t += 1

    def train(self, y):
        """
        バッチ学習
        """
        T = len(y) #データ数(観測数)
        while self.t < T:
            self.update(y[self.t-1])
