import numpy as np
from baumwelch import BaumWelch

class SDHM:
    """
    r: 忘却係数
    nu: 推定係数
    K: HMMの混合数
    n: HMMの次数(n=1)
    M: 総セッション数
    j: j番目
    
    Tj: j番目のセッションの長さ
    N1: 総状態変数
    N2: 総出力シンボル数

    pi: 混合係数
    gamma: 初期確率
    gamma_: gamma導出用
    a: 遷移確率
    a_: a導出用
    b: 出力確率
    b_: b導出用

    tau: t時点で状態siにあり, t+1時点で状態sjにある確率
    tau_: t時点で状態siにある確率
    c: メンバーシップ確率
    """
    def __init__(self, r, nu, K, Tj, N1, N2, n=1):
        """
        r: 忘却係数
        nu: 推定係数
        K: HMMの混合数
        n: HMMの次数(n=1)
        Tj: j番目のセッションの長さ
        N1: 総状態変数
        N2: 総出力シンボル数
        """

        # Given
        self.r = r
        self.nu = nu
        self.K = K
        self.n = n

        # 初期化
        self.Tj = Tj
        self.N1 = N1
        self.N2 = N2

        # パラメータの初期化
        self.pi = None
        self.gamma = None
        self.gamma_ = None
        self.a = None
        self.a_ = None
        self.b = None
        self.b_ = None
        self.c = None
        
        # Baum-Welchで推定
        self.tau = None
        self.tau_ = None

        self.j = 0

    def set_initial_params(self):
        """
        N1: 総状態変数
        N2: 総出力シンボル数

        gamma, gamma_, a, a_, b, b_の初期化
        """

        self.c = np.zeros((1, self.K))
        self.pi = np.ones((1, self.K)) / self.K
        
        #self.gamma = np.zeros((1, self.K, self.N1))
        #self.gamma_ = np.zeros((1, self.K, self.N1))
        #self.a = np.zeros((1, self.K, self.N1, self.N1))
        #self.a_ = np.zeros((1, self.K, self.N1, self.N1))
        #self.b = np.zeros((1, self.K, self.N1, self.N2))
        #self.b_ = np.zeros((1, self.K, self.N1, self.N2))

        self.gamma = np.ones((1, self.K, self.N1)) / self.N1
        self.gamma_ = self.gamma.copy()
        self.a = np.ones((1, self.K, self.N1, self.N1)) / self.N1
        self.a_ = self.a.copy()
        self.b = np.ones((1, self.K, self.N1, self.N2)) / self.N2
        self.b_ = self.b.copy()

        self.tau = np.zeros((1, self.K, self.Tj, self.N1, self.N1))
        self.tau_ = np.zeros((1, self.K, self.Tj, self.N1))

        #for i in range(self.K):
        #    self.gamma = np.zeros((1, self.K, self.N1))
        #    self.gamma_ = np.zeros((1, self.K, self.N1))
        #    self.a = np.zeros((1, self.K, self.N1, self.N1))
        #    self.a_ = np.zeros((1, self.K, self.N1, self.N1))
        #    self.b = np.zeros((1, self.K, self.N1, self.N2))
        #    self.b_ = np.zeros((1, self.K, self.N1, self.N2))

    def add_new_index(self):
        """
        pi, gamma, gamma_, a, a_, b, b_に一行追加
        """
        
        pi_new = np.zeros((1, self.K))
        c_new = np.zeros((1, self.K))
        gamma_new = np.zeros((1, self.K, self.N1))
        gamma_new_ = np.zeros((1, self.K, self.N1))
        a_new = np.zeros((1, self.K, self.N1, self.N1))
        a_new_ = np.zeros((1, self.K, self.N1, self.N1))
        b_new = np.zeros((1, self.K, self.N1, self.N2))
        b_new_ = np.zeros((1, self.K, self.N1, self.N2))
        tau_new = np.zeros((1, self.K, self.Tj, self.N1, self.N1))
        tau_new_ = np.zeros((1, self.K, self.Tj, self.N1))

        self.pi = np.concatenate([self.pi, pi_new])
        self.c = np.concatenate([self.c, c_new])
        self.gamma = np.concatenate([self.gamma, gamma_new])
        self.gamma_ = np.concatenate([self.gamma_, gamma_new_])
        self.a = np.concatenate([self.a, a_new])
        self.a_ = np.concatenate([self.a_, a_new_])
        self.b = np.concatenate([self.b, b_new])
        self.b_ = np.concatenate([self.b_, b_new_])
        self.tau = np.concatenate([self.tau, tau_new])
        self.tau_ = np.concatenate([self.tau_, tau_new_])

    def calc_prob_k(self, alpha):
        """
        k番目のHMMからyjが生成される確率
        """
        
        prob_k = np.sum(alpha[-1])
        return prob_k
    
    def calc_prob(self, pi, prob_k):
        """
        HMMからyjが生成される確率
        """
        
        prob = np.sum(pi * prob_k)
        return prob

    def E_step(self, yj):
        """
        yj: j番目のセッション

        c, tau, tau_の計算
        E-step
        """
        
        prob_k = np.zeros((self.K))
        for k in range(self.K):
            bm = BaumWelch(self.N1, len(yj))
            bm.E_step(yj, self.gamma[self.j, k], self.a[self.j, k], self.b[self.j, k])
            self.tau[self.j, k] = bm.tau
            self.tau_[self.j, k] = bm.tau_
            prob_k[k] = self.calc_prob_k(bm.alpha)
        
        prob = self.calc_prob(self.pi[self.j], prob_k)
        
        self.c[self.j] = (1 - self.nu * self.r) * (prob_k / prob) + self.nu * self.r / self.K

    def M_step(self, yj):
        """
        """
        
        for k in range(self.K):
            self.pi[self.j+1, k] = self.calc_pi(self.r, self.pi[self.j, k], self.c[self.j, k])
            self.gamma_[self.j+1, k] = self.calc_gamma_(self.r, self.N1, self.gamma_[self.j, k], self.tau[self.j, k], c)
            self.gamma[self.j+1, k] = self.calc_gamma(self.Tj, self.N1, self.gamma_[self.j+1, k])
            #self.a_[self.j+1, k] = 

    def calc_pi(self, r, pi, c):
        """
        r: 忘却係数
        pi: 混合係数
        c: メンバーシップ確率
        """

        pi_new = (1 - r) * pi + r * c

        return pi_new

    def calc_gamma_(self, r, N1, gamma_, tau, c):
        """
        r: 忘却係数
        N1: 総状態変数
        gamma_: gamma導出用
        tau: t時点で状態siにあり, t+1時点で状態sjにある確率
        c: メンバーシップ確率
        """

        gamma_new_ = np.zeros((N1))
        for i in range(N1):
            for j in range(N1):
                gamma_new_[i] +=  r * c * tau[1, i, j]
        
        gamma_new_ += (1 - r) * gamma_

        return gamma_new_

    def calc_gamma(self, Tj, N1, gamma_):
        """
        Tj: j番目のセッションの長さ
        N1: 総状態変数
        gamma_: gamma導出用
        """

        gamma_new = np.zeros((N1))
        for i in range(N1):
            gamma_new[i] = gamma_[i] / np.sum(gamma_)

        return gamma_new
  
    def update(self, yj):
        """
        """
        
        if self.j == 0:
            self.set_initial_params()
        self.add_new_index()
        self.E_step(yj)
        #self.M_step()