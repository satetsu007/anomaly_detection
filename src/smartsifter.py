import numpy as np
from sdle import SDLE
from sdem import SDEM

class SmartSifter:
    def __init__(self, r, beta, A, alpha, k, r_h):
        """"""
        # sdle, sdemの共通部初期化
        self.r = r
        # sdleのパラメータ初期化
        self.beta = beta
        self.A = A
        self.M = len(A)
        # sdemのパラメータ初期化
        self.alpha = alpha
        self.k = k
        # sdle, sdemの初期化
        self.sdle = SDLE(self.r, self.beta, self.A)
        self.sdem = [SDEM(self.r, self.alpha, self.k) for i in range(len(self.A))]
        # smartsifterの初期化
        self.t = 1
        # self.p = np.zeros((1, 1)) # p(x, y)を格納
        self.r_h = r_h # ヘリンジャースコアのr
        # スコア格納用
        self.S_L = None # シャノン情報量(対数損失)
        self.S_H = None # ヘリンジャースコア
        
    def update(self, x, y):
        """"""
        # SDLE
        self.sdle.update(x)
        # SDEM: 各セルに対応する混合ガウス分布のパラメータと確率を推定
        for m, A_m in enumerate(self.A):
            if m == int(self.sdle.flag):
                self.sdem[m].update(y)
                # sp_y_x = np.dot(self.sdem[m].prob[self.t], self.sdem[m].pi[self.t])
                # self.p[t, m] = p_y_x * self.sdle.prob[self.t, int(self.sdle.flag)] #t時点における同時確率:p_t(x,y)
            else:
                self.sdem[m].skip(y)
                # self.p[t, m] = self.p[t-1, m]
        # スコアの計算
        s_l = self.calc_logarithmic_loss(y, self.t, self.sdle, self.sdem)
        s_h = [0]
        #s_h = self.calc_hellinger_score(self.sdle.prob[t], self.sdle.prob[t-1], self.sdem, self.t, self.r, self.k, self.M)
        s_l = np.array(s_l)
        s_h = np.array(s_h)
        if self.t ==1:
            self.S_L = s_l
            self.S_H = s_h
        else:
            self.S_L = np.concatenate([self.S_L, s_l])
            self.S_H = np.concatenate([self.S_H, s_h])
            
        self.t += 1
        
    def train(self, x, y, show=False):
        """"""
        T = len(x) # データ数(観測数)
        while self.t <= T:
            self.update(x[self.t-1], y[self.t-1])
            if show:
                if self.t%(T*0.01)==0:
                    print("calculated: " + str(round(self.t/T*100)) + "%")

    def calc_logarithmic_loss(self, y, t, sdle, sdem):
        """
        シャノン情報量(対数損失)を計算
        """
        p_x_prev  = sdle.prob[t-1, sdle.flag]
        pi_prev = sdem[int(sdle.flag)].pi[t-1]
        mu_prev = sdem[int(sdle.flag)].mu[t-1]
        sigma_prev = sdem[int(sdle.flag)].sigma[t-1]
        p_Gauss_prev = sdem[int(sdle.flag)].calc_prob(y, mu_prev, sigma_prev)
        p_y_x_prev = np.dot(pi_prev, p_Gauss_prev)
        p_prev_params = p_x_prev * p_y_x_prev #t-1時点のパラメータによる同時確率:P(x,y)
        s_l = -np.log(p_prev_params) #シャノン情報量を計算
        
        return s_l
        
    def d_h_Gauss(self, mu, sigma, mu_prev, sigma_prev):
        '''
        t時点とt-1時点におけるガウス分布間のヘリンジャー距離
        '''
        #第2項
        m = 2 * np.linalg.det((np.linalg.inv(sigma) + np.linalg.inv(sigma_prev)) / 2) ** (-1/2) #分子
        d = (np.linalg.det(sigma) ** (1/4)) * (np.linalg.det(sigma_prev) ** (1/4)) #分母
        A = m / d

        #第3項 1番目exp
        B_0 = (np.dot(np.linalg.inv(sigma), mu) + np.dot(np.linalg.inv(sigma_prev), mu_prev)).T #要素1
        B_1 = np.linalg.inv(np.linalg.inv(sigma) + np.linalg.inv(sigma_prev)) #要素2
        B_2 = np.dot(np.linalg.inv(sigma), mu) + np.dot(np.linalg.inv(sigma_prev), mu_prev) #要素3
        B = np.exp((1/2) * np.dot(np.dot(B_0, B_1), B_2))

        #第3項 2番目exp
        C_0 = np.dot(np.dot(mu.T, np.linalg.inv(sigma)), mu) + np.dot(np.dot(mu_prev.T, np.linalg.inv(sigma_prev)), mu_prev) #要素
        C = np.exp(-(1/2) * C_0)

        #ヘリンジャー距離の計算
        d_h = 2 - A * B * C

        return d_h

    def d_h_GMM(self, pi, pi_prev, mu, sigma, mu_prev, sigma_prev, k):
        '''
        t時点とt-1時点における混合ガウス分布のヘリンジャー距離
        '''
        d_h = 0
        for i in range(k):
            d_h_G = self.d_h_Gauss(mu[i], sigma[i], mu_prev[i], sigma_prev[i])
            d_h += (np.sqrt(pi[i]) - np.sqrt(pi_prev[i])) ** 2 + (pi[i] + pi_prev[i])/2 * d_h_G

        return d_h

    def calc_hellinger_score(self, p, p_prev, sdem, t, r, k, M):
        '''
        ヘリンジャースコアを計算する関数
        '''
        s_h = 0
        for m in range(M):
            d_h = self.d_h_GMM(sdem[m].pi[t], sdem[m].pi[t-1] ,sdem[m].mu[t], sdem[m].sigma[t], 
                          sdem[m].mu[t-1], sdem[m].sigma[t-1], k)
            tmp = np.sqrt(p[m] * p_prev[m]) #√(p_t(x)*p_t_1(x))      
            s_h += tmp * (d_h - 2)
        s_h += 2
        s_h /= r**2

        return s_h