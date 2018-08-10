# coding: utf-8

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
        Tj: j番目のセッションの長さ
        N1: 総状態変数
        N2: 総出力シンボル数
        n: HMMの次数(n=1)

        パラメータの初期化
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
        #self.pi = self.generate_init_matrix(1, self.K)
        self.pi = np.ones((1, self.K)) / self.K
        
        self.gamma = np.zeros((1, self.K, self.N1))
        self.gamma_ = np.zeros((1, self.K, self.N1))
        self.a = np.zeros((1, self.K, self.N1, self.N1))
        self.a_ = np.zeros((1, self.K, self.N1, self.N1))
        self.b = np.zeros((1, self.K, self.N1, self.N2))
        self.b_ = np.zeros((1, self.K, self.N1, self.N2))

        # 初期化失敗例 (パラメータを全て同一値で初期化)
        #self.gamma = np.ones((1, self.K, self.N1)) / self.N1
        #self.gamma_ = self.gamma.copy()
        #self.a = np.ones((1, self.K, self.N1, self.N1)) / self.N1
        #self.a_ = self.a.copy()
        #self.b = np.ones((1, self.K, self.N1, self.N2)) / self.N2
        #self.b_ = self.b.copy()

        self.tau = np.zeros((1, self.K, self.Tj, self.N1, self.N1))
        self.tau_ = np.zeros((1, self.K, self.Tj, self.N1))

        # generate_init_matrixを使用し, 乱数によりパラメータを初期化
        for i in range(self.K):
            self.gamma[self.j, i] = self.generate_init_matrix(1, self.N1)[0]
            self.gamma_[self.j, i] = self.generate_init_matrix(1, self.N1)[0]
            self.a[self.j, i] = self.generate_init_matrix(self.N1, self.N1)
            self.a_[self.j, i] = self.generate_init_matrix(self.N1, self.N1)
            self.b[self.j, i] = self.generate_init_matrix(self.N1, self.N2)
            self.b_[self.j, i] = self.generate_init_matrix(self.N1, self.N2)

    def generate_init_matrix(self, R, C):
        """
        R: 行数
        C: 列数

        適当な初期値をもつR✕C行列を生成する関数
        """

        # 空の行列を作成
        M = np.zeros((R, C))
        # 適当な値を行ごとに代入していく(各行の合計が1になれば良い)
        for i in range(R):
            rand = np.random.rand(C) #[0, 1]の一様乱数生成
            rand = rand / np.sum(rand) #合計が1になるようにする
            M[i] = rand
        
        return M

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
        alpha: t時点で状態siにあり, その時点までの系列を出力する確率

        k番目のHMMからyjが生成される確率を計算
        """
        
        prob_k = np.sum(alpha[-1])
        return prob_k
    
    def calc_prob(self, pi, prob_k):
        """
        pi: 混合係数
        prob_k: k番目のHMMからyjが生成される確率
        
        HMMからyjが生成される確率
        """
        
        prob = np.dot(pi, prob_k)

        return prob

    def E_step(self, yj):
        """
        yj: j番目のセッション

        c, tau, tau_の計算
        tau, tau_の計算にはBaum-Welchアルゴリズムを使用
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
        for k in range(self.K):
            self.c[self.j, k] = (1 - self.nu * self.r) * (self.pi[self.j, k] * prob_k[k] / prob) + self.nu * self.r / self.K

    def M_step(self, yj):
        """
        yj: j番目のセッション

        pi, gamma, a. bの計算
        M-step
        """
        
        for k in range(self.K):
            # piの更新
            self.pi[self.j+1, k] = (1 - self.r) * self.pi[self.j, k] + self.r * self.c[self.j, k] #OK
            for i in range(self.N1):
                # gamma_の更新
                self.gamma_[self.j+1, k, i] = (1 - self.r) * self.gamma_[self.j, k, i] + self.r * self.c[self.j, k] * np.sum(self.tau[self.j, k, 0, i, :]) #OK
                for j in range(self.N1):
                    # a_の更新
                    self.a_[self.j+1, k, i, j] = (1 - self.r) * self.a_[self.j, k, i, j] + self.r * self.c[self.j, k] * np.sum(self.tau[self.j, k, :-self.n, i, j]) #OK
            for i in range(self.N1):
                # gammaの更新
                self.gamma[self.j+1, k, i] = self.gamma_[self.j+1, k, i] / np.sum(self.gamma_[self.j+1, k]) #OK
                for j in range(self.N1):
                    # aの更新
                    self.a[self.j+1, k, i, j] = self.a_[self.j+1, k, i, j] / np.sum(self.a_[self.j+1, k, i, :]) #OK
            for s in range(self.N1):
                for y in range(self.N2):
                    # b_の更新
                    self.b_[self.j+1, k, s, y] = (1 - self.r) * self.b_[self.j, k, s, y] + self.r * self.c[self.j, k] * np.sum([self.tau_[self.j, k, t, s] for t, yt in enumerate(yj) if y==int(yt)]) #OK
            for s in range(self.N1):
                for y in range(self.N2):
                    # bの更新
                    self.b[self.j+1, k, s, y] = self.b_[self.j+1, k, s, y] / np.sum(self.b_[self.j+1, k, s]) #OK
  
    def update(self, yj):
        """
        yj: j番目のセッション

        オンライン学習
        """
        
        if self.j == 0:
            self.set_initial_params()
        self.add_new_index()
        self.E_step(yj)
        self.M_step(yj)

        self.j += 1

    def train(self, y):
        """
        y: 入力データ(離散値, 複数のセッション)

        バッチ学習
        """

        M = len(y) # 総セッション数
        while self.j < M:
            self.update(y[self.j])
