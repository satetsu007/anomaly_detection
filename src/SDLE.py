import numpy as np

class SDLE:
    def __init__(self, r, beta, A):
        """
        """
        #Given
        self.A = A #セルの分割全体
        self.r = r #忘却係数 
        self.beta = beta #正の定数
        
        
        #パラメータの初期化
        self.M = len(A) #セルの数
        self.A_det = np.zeros(self.M) #各セルに含まれた観測データ数格納用
        self.t = 1
        self.T_t = np.zeros(self.M) #各セルの統計量の計数
        self.prob = np.zeros(self.M)
        self.flag = None #更新の行われたセル番号を格納
    
    def update(self, x_t):
        """
        オンライン学習
        """
        if isinstance(x_t, np.int64):
            x_t = np.array([x_t]) #1次元ベクトルを対応させる
        
        #T_t, probに1行追加
        new_index = np.zeros(self.M)
        if self.t == 1:
            self.T_t = np.concatenate([self.T_t[:,np.newaxis].T, new_index[:,np.newaxis].T])
            self.prob = np.concatenate([self.prob[:,np.newaxis].T, new_index[:,np.newaxis].T])
        
        else:
            self.T_t = np.concatenate([self.T_t, new_index[:,np.newaxis].T])
            self.prob = np.concatenate([self.prob, new_index[:,np.newaxis].T])
        
        for i, A_m in enumerate(self.A):
            delta = 0
            if np.array_equal(x_t, np.array(A_m)):
                delta = 1
                self.A_det[i] += 1
                self.flag = x_t
            
            self.T_t[self.t, i] = (1-self.r) * self.T_t[self.t-1,i] + delta
            if self.r == 0:
                q = (self.T_t[self.t, i] + self.beta) / (self.t + self.M * self.beta)
            else:
                q = (self.T_t[self.t, i]+self.beta) / ((1 - (1-self.r)**self.t)/self.r + self.M*self.beta)
            
            if not self.A_det[i] == 0:
                self.prob[self.t, i] = q / self.A_det[i]
        
        self.t += 1
    
    def train(self, x):
        """
        バッチ学習
        """
        T = len(x) #データ数(観測数)
        while self.t <= T:
            self.update(x[self.t-1])