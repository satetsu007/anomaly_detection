# coding:utf-8

import numpy as np
from baumwelch import BaumWelch
from sdhm import SDHM

class AccessTracer:
    """
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

        self.sdhm = [SDHM(self.r, self.nu, k, self.Tj, self.N1, self.N2) for k in range(1, self.K+1)]
        self.j = 0

        # 対数損失 (Logarithmic Loss)
        self.L_L = None
        # 予測的確率的コンプレキシティ (Predictive Stochastic Complexity)
        self.PSC = None
        # 異常スコア (Score of Anomary)
        self.S_A = None
        # 最適な混合数 (Number of K)
        self.N_K = None

    def set_initial_params(self):
        """
        L_L, PSC, S_A, N_Kの初期化
        """

        self.L_L = np.zeros((1, self.K))
        self.PSC = np.zeros((1, self.K))
        self.S_A = np.zeros(1)
        self.N_K = np.zeros(1)
    
    def add_new_index(self):
        """
        L_L, PSC, S_A, N_Kに一行追加
        """
        
        new_L_L = np.zeros((1, self.K))
        new_PSC = np.zeros((1, self.K))
        new_S_A = np.zeros(1)
        new_N_K = np.zeros(1)

        self.L_L = np.concatenate([self.L_L, new_L_L])
        self.PSC = np.concatenate([self.PSC, new_PSC])
        self.S_A = np.concatenate([self.S_A, new_S_A])
        self.N_K = np.concatenate([self.N_K, new_N_K])
        
    def update(self, yj):
        """
        yj: j番目のセッション
        """

        if self.j == 0:
            self.set_initial_params()
        else:
            self.add_new_index()

        # 固定されたKについて
        for k in range(self.K):
            self.sdhm[k].update(yj)
            prob_k = np.zeros(k+1)
            # HMMm(混合隠れマルコフモデル)の各HMMについて
            for k_ in range(k+1):
                # j-1時点のパラメータを取得
                a = self.sdhm[k].a[self.sdhm[k].j-1, k_]
                b = self.sdhm[k].b[self.sdhm[k].j-1, k_]
                gamma = self.sdhm[k].gamma[self.sdhm[k].j-1, k_]
                # 前向きアルゴリズムの実行
                bm = BaumWelch(self.N1, self.Tj)
                bm.forward(yj, gamma, a, b)
                # prob_kの計算
                prob_k[k_] = self.sdhm[k].calc_prob_k(bm.alpha)
            # probの計算
            prob = self.sdhm[k].calc_prob(self.sdhm[k].pi[self.sdhm[k].j-1], prob_k)
            # 対数損失の計算
            self.L_L[self.j, k] = -np.log(prob)
        # 予測的確率的コンプレキシティの計算
        self.PSC[self.j] = self.L_L[self.j] + self.PSC[self.j-1]
        # 最適な混合数の計算
        self.N_K[self.j] = np.argmin(self.PSC[self.j]) + 1
        # 異常スコアの計算
        self.S_A[self.j] = np.array([self.L_L[self.j, k]/self.Tj for k in range(self.K) if k==self.N_K[self.j]-1])

        self.j += 1

    def all_update(self, y):
        """
        y: 入力データ(離散値, 複数のセッション)

        バッチ学習(オンライン版)
        """

        M = len(y) # 総セッション数
        while self.j < M:
            self.update(y[self.j])