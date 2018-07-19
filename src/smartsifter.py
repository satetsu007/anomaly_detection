import numpy as np

def d_h_Gauss(mu, sigma, mu_prev, sigma_prev):
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


def d_h_GMM(pi, pi_prev, mu, sigma, mu_prev, sigma_prev, k):
    '''
    t時点とt-1時点における混合ガウス分布のヘリンジャー距離
    '''
    d_h = 0
    for i in range(k):
        d_h_G = d_h_Gauss(mu[i], sigma[i], mu_prev[i], sigma_prev[i])
        d_h += (np.sqrt(pi[i]) - np.sqrt(pi_prev[i])) ** 2 + (pi[i] + pi_prev[i])/2 * d_h_G
    
    return d_h


def calc_hellinger_score(p, p_prev, sdem, t, r, k, M):
    '''
    ヘリンジャースコアを計算する関数
    '''
    S_H = 0
    for m in range(M):
        d_h = d_h_GMM(sdem[m].pi[t], sdem[m].pi[t-1] ,sdem[m].mu[t], sdem[m].sigma[t], 
                      sdem[m].mu[t-1], sdem[m].sigma[t-1], k)
        tmp = np.sqrt(p[m] * p_prev[m]) #√(p_t(x)*p_t_1(x))      
        S_H += tmp * (d_h - 2)
    S_H += 2
    S_H /= r**2
    
    return S_H