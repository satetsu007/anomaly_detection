# coding:utf-8
import numpy as np
import numpy.random as rd
import scipy.stats as st
from sdar import SDAR, SDAR_1Dim

class ChangeFinder:
    """
    r: 忘却係数
    T: 第1段階目の平滑化のwindowサイズ
    T_: 第2段階目の平滑化のwindowサイズ
    k: ARモデルの次数

    sdar_f: 第1段階目の学習に使用 (SDARアルゴリズム)
    sdar_s: 第2段階目の学習に使用 (SDARアルゴリズム)

    S_X: 第1段階学習スコア (x_tの対数損失)
    S_T: 第2段階学習スコア (yの対数損失)
    y: 平滑化(移動平均)
    """
    
    def __init__(self, r, T, T_, k):
        """
        r: 忘却係数
        T: 第1段階目の平滑化のwindowサイズ
        T_: 第2段階目の平滑化のwindowサイズ
        k: ARモデルの次数

        パラメータの初期化
        """

        self.r = r
        self.T = T
        self.T_ = T_
        self.k = k

        #self.sdar_f = SDAR(r, k)
        self.sdar_f = SDAR_1Dim(r, k)
        self.sdar_s = SDAR_1Dim(r, k)
        self.w = None

        self.S_X = None
        self.y = None
        self.S_T = None
        self.L_L = None

        self.t = 1

        
    def set_initial_params(self, x_t):
        """
        x_t: t時点のx

        S_X, y, S_T, L_Lの初期化
        """

        self.S_X = np.zeros((1, 1))
        self.y = np.zeros((1, 1))
        self.S_T = np.zeros((1, 1))
        self.L_L = np.zeros((1, 1))


    def add_new_index(self):
        """
        S_X, y, S_T, L_Lに1行追加
        """

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
        x_t: t時点のx

        オンライン学習
        """
        
        if self.t == 1:
            self.set_initial_params(x_t)

        self.add_new_index()
        
        # 第1段階目学習
        self.first_step(x_t)
        # 平滑化
        self.smoothing()
        # 第2段階目学習
        self.second_step()

        self.t += 1

    def p_norm(self, x, mu, sigma):
        """
        x: x
        mu: 平均
        sigma: 分散

        正規分布(mu, sigma)におけるxの確率計算
        """

        p = np.exp(-0.5 *(x-mu)**2/sigma)/((2 * np.pi)**0.5 * sigma**0.5)
        return p
        
    def first_step(self, x_t):
        """
        x: t時点のx

        第1段階目学習
        """

        self.sdar_f.update(x_t)

        # 計算不可(データ不足)
        if self.t < self.k+1:
            self.S_X[self.t] = 0

        else:
            p = self.p_norm(x_t, self.sdar_f.x_t_hat[self.t-1], self.sdar_f.sigma_hat[self.t-1])
            #p = st.multivariate_normal.pdf(x_t, self.sdar_f.x_t_hat[self.t-1], self.sdar_f.sigma_hat[self.t-1])
            self.S_X[self.t] = -np.log(p + 1e-11)

    def smoothing(self):
        """
        平滑化
        """

        # 計算可能条件
        if len(self.S_X) > self.k + self.T:
            self.y[self.t] = np.mean(self.S_X[self.t-self.T+1:self.t+1])

        
    def second_step(self):
        """
        第2段階目学習, 平滑化
        """


        # 対数損失を計算する
        # SDAR更新不可
        if len(self.S_X) <= self.k + self.T:
            self.L_L[self.t] = 0
            #self.S_T[self.t] = 0

        # SDAR更新可
        else:
            self.sdar_s.update(self.y[self.t])
            
            # 対数損失計算可
            if len(self.sdar_s.x) > self.k + 1:
                # probを求める
                #p = st.norm.pdf(self.y[self.t], self.sdar_s.x_t_hat[-1], self.sdar_s.sigma_hat[-1])
                p = self.p_norm(self.y[self.t], self.sdar_s.x_t_hat[-1], self.sdar_s.sigma_hat[-1])
            
                self.L_L[self.t] = -np.log(p+1e-11)

            else:
                self.L_L[self.t] = 0
            
        # 平均スコア算出不可
        if self.t < 2 * self.T + self.T_:
            self.S_T[self.t] = 0

        # 平均スコア算出可
        else:
            self.S_T[self.t] = np.mean(self.L_L[self.t-self.T_+1:self.t+1])