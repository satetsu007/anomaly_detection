# coding:utf-8
import numpy as np
import numpy.random as rd
import scipy.stats as st
from sdar import SDAR, SDAR_1Dim

class ChangeFinder:
    """
    r
    T
    T_

    S_X: 第一段階学習スコア (x_tの対数損失)
    S_T: 第二段階学習スコア (yの対数損失)
    y: 平滑化(移動平均)
    """
    
    def __init__(self, r, T, T_, k):
        """
        r
        T
        T_
        k: ARモデルの次数
        """

        self.r = r
        self.T = T
        self.T_ = T_
        self.k = k

        self.sdar_f = SDAR(k, r)
        #self.sdar_f = SDAR_1Dim(k, r)
        self.sdar_s = SDAR_1Dim(k, r)
        self.w = None

        self.S_X = None
        self.y = None
        self.S_T = None
        self.L_L = None

        self.t = 1

        
    def set_initial_params(self, x_t):
        '''
        '''

        self.S_X = np.zeros((1, 1))
        self.y = np.zeros((1, 1))
        self.S_T = np.zeros((1, 1))
        self.L_L = np.zeros((1, 1))


    def add_new_index(self):
        '''
        '''

        S_X_new = np.zeros((1, 1))
        y_new = np.zeros((1, 1))
        S_T_new = np.zeros((1, 1))
        L_L_new = np.zeros((1, 1))

        self.S_X = np.concatenate([self.S_X, S_X_new])
        self.y = np.concatenate([self.y, y_new]) 
        self.S_T = np.concatenate([self.S_T, S_T_new])
        self.L_L = np.concatenate([self.L_L, L_L_new])
    
    def update(self, x_t):
        """
        """
        
        if self.t == 1:
            self.set_initial_params(x_t)

        self.add_new_index()
        
        self.first_step(x_t)
        self.smoothing()
        self.second_step()

        self.t += 1
        
    def first_step(self, x_t):
        """
        x: t時点のx
        """

        self.sdar_f.update(x_t)
        #p = st.norm.pdf(x_t, self.sdar_f.x_t_hat[self.t-1], self.sdar_f.sigma_hat[self.t-1])
        p = st.multivariate_normal.pdf(x_t, self.sdar_f.x_t_hat[self.t-1], self.sdar_f.sigma_hat[self.t-1])
        self.S_X[self.t] = -np.log(p + 1e-6)

    def smoothing(self):
        """
        """

        if self.t >= self.T:
            self.y[self.t] = np.mean(self.S_X[self.t-self.T+1:self.t+1])

        
    def second_step(self):
        """
        """

        self.sdar_s.update(self.y[self.t])
        
        p = st.norm.pdf(self.y[self.t], self.sdar_s.x_t_hat[self.t-1], self.sdar_s.sigma_hat[self.t-1])
        self.L_L[self.t] = -np.log(p + 1e-6)
        
        if self.t >= self.T_:
            self.S_T[self.t] = np.mean(self.L_L[self.t-self.T_+1:self.t+1])