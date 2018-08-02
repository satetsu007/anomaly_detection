# coding: utf-8

import numpy as np

class BaumWelch:
    """
    yj: j番目のセッション
    N1: 総状態変数
    Tj: j番目のセッションの長さ

    gamma: 初期確率
    a: 遷移確率
    b: 出力確率

    alpha: t時点で状態siにあり, その時点までの系列を出力する確率
    beta: t時点で状態siにいるとして, t+1時点から最後までの系列を出力する確率
    tau: t時点で状態siにあり, t+1時点で状態sjにある確率
    tau_: t時点で状態siにある確率
    """
    def __init__(self, N1, Tj):
        """
        N1: 総状態変数
        Tj: j番目のセッションの長さ

        alpha, beta, tau, tau_ の初期化
        """

        self.N1 = N1
        self.Tj = Tj
        # 格納用行列を生成
        self.alpha = np.zeros((Tj, N1))
        self.beta = np.zeros((Tj, N1))
        self.tau = np.zeros((Tj, N1, N1))
        self.tau_ = np.zeros((Tj, N1))

    def forward(self, yj, gamma, a, b):
        """
        yj: j番目のセッション
        gamma: 初期確率
        a: 遷移確率
        b: 出力確率
        
        前向きアルゴリズム
        αの計算
        """
        # t=0時点のalpha
        self.alpha[0] = gamma * b[:, yj[0]]

        # 再帰計算
        for t in range(self.Tj-1):
            for j in range(self.N1):
                self.alpha[t+1, j] = np.sum(self.alpha[t] * a[:, j]) * b[j, yj[t+1]]

    def backward(self, yj, a, b):
        """
        yj: j番目のセッション
        a: 遷移確率
        b: 出力確率
        
        後ろ向きアルゴリズム
        βの計算
        """
        #βについて(backward推定)
        #Step1: 初期化
        self.beta[-1] = 1

        #Step2: 再帰的計算
        for t in range(self.Tj-1)[::-1]:
            for i in range(self.N1):
                self.beta[t, i] = np.sum(a[i] * b[:, yj[t+1]] * self.beta[t+1])
    
    def calc_tau(self, yj, a, b):
        """
        yj: j番目のセッション
        a: 遷移確率
        b: 出力確率

        τの計算
        """
        for t in range(self.Tj-1):
            #分母(denominator)を計算する
            d = np.sum([np.sum(self.alpha[t, i] * a[i, :] * b[:, yj[t+1]] * self.beta[t+1, :]) for i in range(self.N1)])

            for i in range(self.N1):
                for j in range(self.N1):
                    self.tau[t, i, j] = self.alpha[t, i] * a[i, j] * b[j, yj[t+1]] * self.beta[t+1, j] / d

    def calc_tau_(self):
        """
        τ'の計算
        """
        #τ'について
        for t in range(self.Tj):
            for i in range(self.N1):
                self.tau_[t, i] = np.sum(self.tau[t, i, :])

    def E_step(self, yj, gamma, a, b):
        """
        yj: j番目のセッション
        gamma: 初期確率
        a: 遷移確率
        b: 出力確率
        
        α, β, τ, τ'の計算
        """

        self.forward(yj, gamma, a, b)
        self.backward(yj, a, b)
        self.calc_tau(yj, a, b)
        self.calc_tau_()