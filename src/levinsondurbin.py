#-*- coding:utf-8 -*-
import numpy as np


class LevinsonDurbin:
    '''
    レヴィンソン型のアルゴリズム
    C: 自己共分散関数(j=1,...,k)
    k: ARモデルの次数
    d: データの次元
    '''
    def __init__(self, C, k):
        # 初期化
        self.C = C
        self.k = k
        self.d = C.shape[1]

        # 計算結果格納用
        self.A = np.zeros((k+1, k+1, self.d, self.d))
        self.B = np.zeros((k+1, k+1, self.d, self.d))
        self.E = np.zeros(C.shape)
        self.W = np.zeros(C.shape)
        self.Z = np.zeros(C.shape)


        # 初期値を設定する
        self.W[0] = C[0]
        self.Z[0] = C[0]

        # 推定係数格納用
        self.coeffs = None

    def estimate(self, m):
        '''
        m次のモデルの係数を推定する(※ m > 0)
        '''

        self.E[m] = self.C[m]
        if m-1 > 0:
            for i in range(m-1):
                self.E[m] -= np.dot(self.A[m-1, i+1], self.C[m-i-1])

        self.A[m, m] = np.dot(self.E[m], np.linalg.inv(self.Z[m-1]))
        self.B[m, m] = np.dot(self.E[m].T, np.linalg.inv(self.W[m-1]))

        for i in range(m-1):
            self.A[m, i+1] = self.A[m-1, i+1] - np.dot(self.A[m, m], self.B[m-1, m-i-1])
            self.B[m, i+1] = self.B[m-1, i+1] - np.dot(self.B[m, m], self.A[m-1, m-i-1])

        self.W[m] = self.C[0]
        self.Z[m] = self.C[0]
        for i in range(m):
            self.W[m] -= np.dot(self.A[m, i+1], self.C[i+1].T)
            self.Z[m] -= np.dot(self.B[m, i+1], self.C[i+1])

    def solve_YW(self):
        '''
        Yule-Walker方程式を解く
        '''
        for i in range(self.k):
            self.estimate(i+1)

        self.coeffs = self.A[-1]

def LevinsonDurbin_1Dim(r, k):
    """
    from http://aidiary.hatenablog.com/entry/20120415/1334458954
    """
    a = np.zeros(k+1, dtype=np.float64)
    e = np.zeros(k+1, dtype=np.float64)

    a[0] = 1.0
    a[1] = - r[1] / r[0]
    e[1] = r[0] + r[1] * a[1]
    lam = - r[1] / r[0]

    for k in range(1, k):
        lam = 0.0
        for j in range(k+1):
            lam -= a[j] * r[k+1-j]
        lam /= e[k]

        U = [1]
        U.extend([a[i] for i in range(1, k+1)])
        U.append(0)

        V = [0]
        V.extend([a[i] for i in range(k, 0, -1)])
        V.append(1)

        a = np.array(U) + lam * np.array(V)
        e[k + 1] = e[k] * (1.0 - lam * lam)

    return a, e[-1]