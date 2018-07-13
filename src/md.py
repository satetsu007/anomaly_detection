import numpy as np

class mahalanobis_distance:
    def __init__(self, x):
        self.mu = np.mean(x, axis=0)
        self.sigma = np.cov(x.T)
        self.sort_index = np.zeros(x.shape[0])
        self.x_md = np.zeros((x.shape[0], x.shape[1]+1))
        self.flag = np.zeros(x.shape[0])
        self.x_flag = np.zeros((x.shape[0], x.shape[1]+1))
        
    def calc_distance(self, x):
        """
        マハラノビス距離の算出
        """
        md = np.zeros((x.shape[0]))
        for i, ix in enumerate(x):
            md[i] = np.sqrt(np.dot(np.dot((ix - self.mu), np.linalg.inv(self.sigma)), (ix - self.mu)))
        
        self.md = md
        
    def sort_distance(self, x):
        """
        マハラノビス距離に基づき並び替える
        """
        self.sort_index = np.argsort(self.md)
        self.x_md = np.concatenate((x, self.sort_index.reshape(x.shape[0], 1)), axis=1)
        self.x_md[:, -1].sort()
    
    def check_outlier(self, theta):
        """
        thetaを基準に外れ値検出を行う
        """
        flag = []
        for i, imd in enumerate(self.md):
            if imd > theta:
                flag.append(1)
            else:
                flag.append(0)
        self.flag = np.array(flag)
        self.x_flag = np.concatenate((x, self.flag.reshape(self.md.shape[0], 1)), axis=1)