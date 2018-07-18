import numpy as np

def hellinger_distance(mu, sigma, mu_prev, sigma_prev):
    mu = mu.reshape((mu.shape[0], 1))
    mu_prev = mu_prev.reshape((mu_prev.shape[0], 1))
    
    d_h = -(2 * np.linalg.det((np.linalg.inv(sigma) + np.linalg.inv(sigma_prev)) / 2) ** (-1/2)) / (np.linalg.det(sigma) ** (1/4) * np.linalg.det(sigma_prev) ** (1/4))
    
    #tmp = np.exp((1/2) * np.dot(np.dot(np.dot(np.linalg.inv(sigma), mu) + np.dot(np.linalg.inv(sigma_prev), mu_prev).T, 
    #                          np.linalg.inv(np.linalg.inv(sigma) + np.linalg.inv(sigma_prev))),
    #                          np.dot(np.linalg.inv(sigma), mu + np.dot(np.linalg.inv(sigma_prev), mu_prev).T)))
    A = (np.dot(np.linalg.inv(sigma), mu) + np.dot(np.linalg.inv(sigma_prev), mu_prev)).T
    B = np.linalg.inv(np.linalg.inv(sigma) + np.linalg.inv(sigma_prev))
    C = np.dot(np.linalg.inv(sigma), mu) + np.dot(np.linalg.inv(sigma_prev), mu_prev)
    
    #print("A", A)
    #print("B", B)
    #print("C", C)
    tmp = np.exp((1/2) * np.dot(np.dot(A, B), C))
    tmp *= np.exp(-(1/2) * (np.dot(np.dot(mu.T, np.linalg.inv(sigma)), mu) + np.dot(np.dot(mu_prev.T, np.linalg.inv(sigma_prev)), mu_prev)))
    d_h *= tmp
    d_h += 2
    return d_h

def calc_hellinger_distance(pi, pi_prev, mu, sigma, mu_prev, sigma_prev, k):
    d_H = 0
    for i in range(k):
        d_h = hellinger_distance(mu[i], sigma[i], mu_prev[i], sigma_prev[i])
        d_H += (pi[i] + pi_prev[i])/2 * d_h
    tmp = 0
    for i in range(k):
        tmp += np.sqrt(pi[i]) - np.sqrt(pi_prev[i]) ** 2
    d_H += tmp 
    return d_H

def calc_hellinger_score(p, p_prev, sdem, r, k, M):
    S_H = 0
    for m in range(M):
        d_H = calc_hellinger_distance(sdem[m].pi[sdem[m].t-1], sdem[m].pi[sdem[m].t-2] ,sdem[m].mu[sdem[m].t-1],
                                      sdem[m].sigma[sdem[m].t-1], sdem[m].mu[sdem[m].t-2], sdem[m].sigma[sdem[m].t-2], k)
        S_H += np.sqrt(p[m] * p_prev[m]) * d_H
    tmp = 0
    for m in range(M):
        tmp += np.sqrt(p[m] * p_prev[m])
    tmp *= 2
    S_H += 2 - tmp
    S_H /= r**2
    
    return S_H