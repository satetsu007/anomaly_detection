#-*- coding:utf-8 -*-
import numpy as np
import numpy.random as rd
from levinsondurbin import LevinsonDurbin, LevinsonDurbin_1Dim

class SDAR:
    """
    r
    k
    x
    d
    t
    mu_hat
    C
    x_t_hat
    omega_hat
    sigma_hat
    """
    def __init__(self, r, k):
        """
        r: 忘却係数
        k: ARモデルの次数

        パラメータの初期化
        """
        # Given
        self.r = r # 忘却係数
        self.k = k # ARモデルの次数

        # 初期化
        self.x = None # 入力されたデータ
        self.d = None #入力データの次元
        self.t = 1 # 時点

        # パラメータの初期化
        self.mu_hat = None
        self.C = None
        self.x_t_hat = None
        self.omega_hat = None
        self.sigma_hat = None

    def update(self, x_t):
        """
        x_t: t時点のx

        オンライン学習
        """

        if self.t == 1:
            self.set_initial_params(x_t)

        self.add_new_index()

        # x_t を格納
        self.x[self.t] = x_t

        # μ^の更新
        self.mu_hat[self.t] = (1 - self.r) * self.mu_hat[self.t-1] + self.r * x_t

        # Cの更新
        for j in range(self.k+1):
            if self.t > j:
                self.C[self.t, j] = (1 - self.r) * self.C[self.t-1, j] + self.r \
                                * np.dot((x_t - self.mu_hat[self.t])[:, np.newaxis], (self.x[self.t-j] - self.mu_hat[self.t])[:, np.newaxis].T)


        if self.t > self.k:
            # Yule-Walker方程式を解く
            #print(self.C[-1])
            ld = LevinsonDurbin(self.C[self.t], self.k)
            ld.solve_YW()
            self.omega_hat[self.t] = ld.coeffs

            # x_t^を計算する
            self.x_t_hat[self.t] = self.mu_hat[self.t]
            for i in range(self.k):
                vector = self.x[self.t-i-1] - self.mu_hat[self.t]
                self.x_t_hat[self.t] += np.dot(self.omega_hat[self.t, i+1], vector)

            # Σ^を計算する
            vector2 = (self.x[self.t] - self.x_t_hat[self.t])[:, np.newaxis]
            self.sigma_hat[self.t] = (1-self.r) * self.sigma_hat[self.t-1] + self.r * np.dot(vector2, vector2.T)


        else:
            # 前の時点の値を引き継ぐ
            self.sigma_hat[self.t] = self.sigma_hat[self.t-1]



        self.t += 1


    def set_initial_params(self, x_t):
        """
        x_t: t時点のx

        d, x, x_t_hat, mu_hat, C, omega_hat, sigma_hatの初期化
        """

        self.d = x_t.shape[0] # 入力データの次元を取得
        self.x = np.zeros((1, self.d)) # 入力データ格納用
        self.x_t_hat = np.zeros((1, self.d)) # x_t^の初期化
        self.mu_hat = rd.uniform(low=0, high=1, size=self.d)[np.newaxis, :] # μ^の初期化:乱数
        self.C = np.zeros((1, self.k+1, self.d, self.d)) # C_jの初期化: 全てゼロ
        self.omega_hat = np.zeros((1, self.k+1, self.d, self.d)) # ω^の初期化: 全てゼロ
        self.sigma_hat = np.eye(self.d)[np.newaxis, :] # Σ^の初期化: 単位行列

    def add_new_index(self):
        """
        x, x_t_hat, mu_hat, C, omega_hat, sigma_hatに1行追加
        """

        x_new = np.zeros((1, self.d))
        mu_new = np.zeros((1, self.d))
        C_new = np.zeros((1, self.k+1, self.d, self.d))
        x_t_hat_new = np.zeros((1, self.d))
        omega_hat_new = np.zeros((1, self.k+1, self.d, self.d))
        sigma_hat_new = np.zeros((1, self.d, self.d))

        self.x = np.concatenate([self.x, x_new])
        self.mu_hat = np.concatenate([self.mu_hat, mu_new])
        self.C = np.concatenate([self.C, C_new])
        self.x_t_hat = np.concatenate([self.x_t_hat, x_t_hat_new])
        self.omega_hat = np.concatenate([self.omega_hat, omega_hat_new])
        self.sigma_hat = np.concatenate([self.sigma_hat, sigma_hat_new])


class SDAR_1Dim:
    """
    """
    def __init__(self, r, k):
        """
        r: 忘却係数
        k: ARモデルの次数
        """

        # Given
        self.r = r # 忘却係数
        self.k = k # ARモデルの次数

        # 初期化
        self.mu_hat = None
        self.C = None
        self.x_t_hat = None
        self.omega_hat = None
        self.sigma_hat = None

        self.x = None # 入力されたデータ
        self.t = 1 # 時点

    def update(self, x_t):
        """
        """

        if self.t == 1:
            self.set_initial_params(x_t)

        self.add_new_index()

        # x_t を格納
        self.x[self.t] = x_t

        # 更新しないで、前の時点の値を引き継ぐ
        if self.t < self.k+1:
            self.mu_hat[self.t] = self.mu_hat[self.t-1]
            self.sigma_hat[self.t] = self.sigma_hat[self.t-1]

        else:
            # μ^の更新
            self.mu_hat[self.t] = (1 - self.r) * self.mu_hat[self.t-1] + self.r * x_t

            # Cの更新
            for j in range(self.k+1):
                    self.C[self.t, j] = (1 - self.r) * self.C[self.t-1, j] + self.r \
                                    * (x_t - self.mu_hat[self.t]) * (self.x[self.t-j] - self.mu_hat[self.t])

            # Yule-Waker方程式を解く
            self.omega_hat[self.t], _ = LevinsonDurbin_1Dim(self.C[self.t], self.k)
            self.omega_hat[self.t] = - self.omega_hat[self.t]
            
            # x^の計算
            self.x_t_hat[self.t] = self.mu_hat[self.t]
            for i in range(self.k):
                self.x_t_hat[self.t] += self.omega_hat[self.t, i+1] * (self.x[self.t-i-1] - self.mu_hat[self.t])
            
            # σの計算
            self.sigma_hat[self.t] = (1 - self.r) * self.sigma_hat[self.t-1] + \
                                self.r * ((x_t - self.x_t_hat[self.t]) ** 2)

        self.t += 1

    def set_initial_params(self, x_t):
        '''
        '''

        self.x = np.zeros((1, 1))
        self.x_t_hat = np.zeros((1, 1))
        self.C = np.zeros((1, self.k+1))
        # self.C = (np.random.random(self.k+1) / 100.0)[np.newaxis, :]
        self.mu_hat = rd.uniform(low=0, high=1, size=1)[:, np.newaxis]
        self.sigma_hat = rd.uniform(low=0, high=1, size=1)[:, np.newaxis]
        self.omega_hat = np.zeros((1, self.k+1))


    def add_new_index(self):
        '''
        '''

        x_new = np.zeros((1, 1))
        x_t_hat_new = np.zeros((1, 1))
        C_new = np.zeros((1, self.k+1))
        mu_hat_new = np.zeros((1, 1))
        sigma_hat_new = np.zeros((1, 1))
        omega_hat_new = np.zeros((1, self.k+1))

        self.x = np.concatenate([self.x, x_new])
        self.x_t_hat = np.concatenate([self.x_t_hat, x_t_hat_new])
        self.C = np.concatenate([self.C, C_new])
        self.mu_hat = np.concatenate([self.mu_hat, mu_hat_new])
        self.sigma_hat = np.concatenate([self.sigma_hat, sigma_hat_new])
        self.omega_hat = np.concatenate([self.omega_hat, omega_hat_new])
