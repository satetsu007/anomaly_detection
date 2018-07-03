import  os
import  numpy as np
import  numpy.random  as rd
import scipy as sp
from scipy import stats as st

class EM:
    def __init__(self, data, K):
        """
        """
        D = data.shape[1]
        N = data.shape[0]
        # initialize pi
        pi = np.zeros(K)
        for k in range(K):
            if k == K-1:
                pi[k] = 1 - np.sum(pi)
            else:
                pi[k] = 1/K

        # initialize mu
        max_x, min_x = np.max(data[:,0]), np.min(data[:,0])
        max_y, min_y = np.max(data[:,1]), np.min(data[:,1])
        mu = np.c_[rd.uniform(low=min_x, high=max_x, size=K), rd.uniform(low=min_y, high=max_y, size=K)]

        # initialize sigma
        # sigma = np.asanyarray([np.eye(D) * 0.1 for k in range(K)])
        sigma = np.asanyarray([np.eye(D) for k in range(K)])

        self.K = K
        self.N = N
        self.D = D
        self.pi = pi
        self.mu = mu
        self.sigma = sigma
        # calculate likelihood
        likelihood = self.calc_likelihood(data)
        self.likelihood = likelihood
        self.t = 0

    def E_step(self):
        """
        """
        gamma = (self.likelihood.T / np.sum(self.likelihood, axis=1)).T
        self.gamma = gamma

    def M_step(self, data):
        """
        """
        # caluculate pi
        N_k = np.array([np.sum(self.gamma[:,k]) for k in range(self.K)])
        pi = N_k / self.N

        # calculate mu
        tmp_mu = np.zeros((self.K, self.D))

        for k in range(self.K):
            for i in range(len(data)):
                tmp_mu[k] += self.gamma[i, k]*data[i]
            tmp_mu[k] = tmp_mu[k]/N_k[k]
        mu_prev = self.mu.copy()
        mu = tmp_mu.copy()
        
        # calculate sigma
        tmp_sigma = np.zeros((self.K, self.D, self.D))

        for k in range(self.K):
            tmp_sigma[k] = np.zeros((self.D, self.D))
            for i in range(self.N):
                tmp = np.asanyarray(data[i]-mu[k])[:,np.newaxis]
                tmp_sigma[k] += self.gamma[i, k]*np.dot(tmp, tmp.T)
            tmp_sigma[k] = tmp_sigma[k]/N_k[k]

        sigma = tmp_sigma.copy()

        self.mu = mu
        self.sigma = sigma
        self.pi = pi
        self.mu_prev = mu_prev

    def convergence_check(self, data):
        """
        """
        # calculate likelihood
        prev_likelihood = self.likelihood
        self.likelihood = self.calc_likelihood(data)
        
        prev_sum_log_likelihood = np.sum(np.log(prev_likelihood))
        sum_log_likelihood = np.sum(np.log(self.likelihood))
        self.diff = prev_sum_log_likelihood - sum_log_likelihood
        self.t += 1

    def EM_Algorithm(self, data, prm):
        # EM-Algorithm
        while(True):
            # E-step
            self.E_step()
            # M-step
            self.M_step(data)
            # convergence-check
            self.convergence_check(data)
            if(abs(self.diff)<prm):
                break

    def calc_likelihood(self, data):
        likelihood = np.zeros((self.N, self.K))
        for k in range(self.K):
            likelihood[:, k] = [self.pi[k]*st.multivariate_normal.pdf(d, self.mu[k], self.sigma[k]) for d in data]
        return likelihood