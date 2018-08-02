# coding: utf-8

import numpy as np

class mahalanobis_distance:
    """
    mu: 期待値
    sigma: 分散共分散行列
    sort_index: ソート時のindex
    x_md: xのマハラノビス距離
    flag: 外れ値か判別するフラグ
    x_flag: xが外れ値か判別するフラグ
    """
    def __init__(self, x):
        """
        x: 入力データ

        初期化
        """
        self.mu = np.mean(x, axis=0)
        self.sigma = np.cov(x.T)
        self.sort_index = np.zeros(x.shape[0])
        self.x_md = np.zeros((x.shape[0], x.shape[1]+1))
        self.flag = np.zeros(x.shape[0])
        self.x_flag = np.zeros((x.shape[0], x.shape[1]+1))
        
    def calc_distance(self, x):
        """
        x: 入力データ

        マハラノビス距離の算出
        """
        md = np.zeros((x.shape[0]))
        for i, ix in enumerate(x):
            if len(ix) == 1:
                md[i] = np.sqrt((ix - self.mu)*(1/self.sigma)*(ix - self.mu))
            else:
                md[i] = np.sqrt(np.dot(np.dot((ix - self.mu), np.linalg.inv(self.sigma)), (ix - self.mu)))
        
        self.md = md
        
    def sort_distance(self, x):
        """
        x: 入力データ

        マハラノビス距離に基づき並び替える
        """
        self.sort_index = np.argsort(self.md)
        self.x_md = np.concatenate((x, self.sort_index.reshape(x.shape[0], 1)), axis=1)
        self.x_md[:, -1].sort()
    
    def check_outlier(self, x, theta):
        """
        x: 入力データ
        theta: 閾値

        thetaを基準に外れ値検出を行う
        """
        flag = []
        for imd in enumerate(self.md):
            if imd > theta:
                flag.append(1)
            else:
                flag.append(0)
        self.flag = np.array(flag)
        self.x_flag = np.concatenate((x, self.flag.reshape(self.md.shape[0], 1)), axis=1)