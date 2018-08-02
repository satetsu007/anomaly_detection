# coding: utf-8

import os
import numpy as np
import numpy.random as rd
import pandas as pd
from scipy import stats as st
import math
from crp import CRP


class DPM:
    def __init__(self, x, alpha, mu0, beta, nu, S):
        '''
        x: データ
        alpha: クラスタを分割しやすくするパラメータ. 大きいほど, 分割されやすくなる.
        mu0: クラスタの平均に影響を与えるパラメータ. betaの値が小さいひどほとんど影響しない.
        beta: 各クラスの平均, 分散, 分割のされやすさに影響を与えるパラメータ.
        nu,S: nu,Sが大きいほど、クラスタが分割されやすくなる.
        '''
        self.x = x
        self.num_samples = len(x)

        #ハイパーパラメータ
        self.alpha = alpha
        self.mu0 = mu0
        self.beta = beta
        self.nu = nu
        self.S = S

        #初期設定
        self.s, self.n_ = CRP(alpha, self.num_samples) #潜在変数と各クラスタに所属する総数
        self.s_hat = self.s.copy() #推定した潜在変数
        self.n_hat = self.n_.copy() #推定した各クラスタに所属する総数
        self.c = len(self.n_.keys()) #クラスタ数
        self.c_hat = self.c #更新毎の推定したクラスタ数
        self.d = x.shape[1] #データの次元

        #θ(mu, Lambda)の初期化
        self.mu = {}
        self.Lambda = {}
        for i in range(self.c):
            mu_i, Lambda_i = self.calc_params(i, x, self.s, self.n_, mu0, beta, nu, S)
            self.mu[i] = mu_i
            self.Lambda[i] = Lambda_i

        self.mu_hat = self.mu.copy() #推定したパラーメ:μ
        self.Lambda_hat = self.Lambda.copy() #推定したパラメータ: Λ

        #計算結果格納用
        self.log_post_prob = self.calc_post_prob(x, self.s, self.c, self.n_, self.mu, self.Lambda, alpha, beta, self.num_samples, mu0, nu, S)
        self.prob_max = self.log_post_prob #事後確率最大用


    def Gibbs_sampling(self):
        '''
        ギブスサンプリングを実行する
        '''
        xs = np.c_[self.x, self.s]

        #任意の1つのデータを選択
        del_num = rd.randint(self.num_samples)

        #該当するデータを抽出
        del_xs =  xs[del_num]
        del_x = del_xs[:2].copy()
        del_s = int(del_xs[2].copy())

        #該当するクラスタの総数を1減らす
        self.n_[del_s] -= 1

        #空きクラスタが発生した場合は、クラスタ数とパラメータのインデックスを更新する
        if self.n_[del_s] == 0:
            self.c -= 1
            del self.n_[del_s]
            del self.mu[del_s]
            del self.Lambda[del_s]
            new_keys = [i for i in range(self.c)]
            self.n_ = {key:value for key, value in zip(new_keys, self.n_.values())}
            self.mu = {key:value for key, value in zip(new_keys, self.mu.values())}
            self.Lambda = {key:value for key, value in zip(new_keys, self.Lambda.values())}
            self.s = [s - 1 if s > del_s else s for s in self.s]

        #s_kの値を確率的に決める
        p_s_k = np.zeros(self.c + 1)
        #既存の各クラスに対して確率(p(s_k))を計算する
        for i in range(self.c):
            prob = (self.n_[i]/(self.num_samples - 1 + self.alpha)) \
                    * st.multivariate_normal.pdf(del_x, self.mu[i], np.linalg.inv(self.Lambda[i]))
            p_s_k[i] = prob

        #新規のクラスタに対して確率を計算する
        #基底分布(正規-ウィシャート分布)として確率を計算する
        #計算に必要な要素を計算する
        xk_mu0 = (del_x - self.mu0)[:, np.newaxis]
        S_b_inv = np.linalg.inv(self.S) + self.beta/(1+self.beta) * np.dot(xk_mu0, xk_mu0.T)
        S_b = np.linalg.inv(S_b_inv)
        A = (self.beta / ((1 + self.beta) * np.pi)) ** (self.d/2)
        B = np.linalg.det(S_b)**((self.nu+1)/2) * math.gamma((self.nu+1)/2)
        C = np.linalg.det(self.S)**(self.nu/2) * math.gamma((self.nu+1-self.d)/2)

        #確率を計算する
        p_tmp = A * B / C #p269 式(12.35)
        prob = self.alpha / (self.num_samples - 1 + self.alpha) * p_tmp
        p_s_k[-1] = prob
        p_s_k = p_s_k / np.sum(p_s_k) #正規化

        #s_kの値を確率的にクラスタを決める
        judge = rd.random() #0-1の範囲で乱数を取得
        p_sum = 0
        s_k = 0
        for i, p in enumerate(p_s_k):
            p_sum += p
            if judge < p_sum:
                s_k = i
                break

        #s_kの値を更新する
        self.s[del_num] = s_k

        #新規クラスタが発生した場合は、総クラスタ数と所属パターン数, パラメータを更新する
        if s_k == self.c:
            self.n_[s_k] = 1
            self.c += 1
            mu_new, Lambda_new = self.calc_params(s_k, self.x, self.s, self.n_, self.mu0, self.beta, self.nu, self.S)
            self.mu[s_k] = mu_new
            self.Lambda[s_k] = Lambda_new

        #既存のクラスタに更新された場合は、所属パターン数を1増やす
        else:
            self.n_[s_k] += 1

        #パラメータθ(μ,Λ)の更新(更新があったクラスタのみ)
        #*注意*:空きクラスタになる→既存クラスに所属→空きクラスタ(del_s)のパラメータは更新しない
        if del_s != s_k:
            for c_num in [del_s, s_k]:
                if del_s not in self.n_.keys():
                    continue
                mu_i, Lambda_i = self.calc_params(c_num, self.x, self.s, self.n_, self.mu0, self.beta, self.nu, self.S)
                self.mu[c_num] = mu_i
                self.Lambda[c_num] = Lambda_i

        #事後確率:p(s,θ|x)(対数)を求める
        self.log_post_prob = self.calc_post_prob(self.x, self.s, self.c, self.n_, self.mu, self.Lambda, self.alpha, self.beta, self.num_samples, self.mu0, self.nu, self.S)


    def train(self, min_step=10000, stop_count=1000, verbose=0):
        '''
        #ギブスサンプリングを繰り返し、更新されない状態が続いた時打ち切る
        #最低限の繰り返し回数
        #stop_count:更新されない回数が何回続いたら処理を打ち切るか指定するパラメータ
        #verbose: 1の時、ステップ回数を表示
        '''
        update = True
        count = 0
        step = 1
        while update:
            if verbose:
                print("step:", step)
            #ギブスサンプリングを実行する
            self.Gibbs_sampling()

            #最低限の繰り返し処理
            if step < min_step:
                if self.prob_max <= self.log_post_prob:
                    #更新する
                    self.prob_max = self.log_post_prob.copy()
                    self.s_hat = self.s.copy()
                    self.c_hat = self.c
                    self.mu_hat = self.mu.copy()
                    self.Lambda_hat = self.Lambda.copy()
                    self.n_hat = self.n_.copy()
                step += 1
                continue

            #処理の打ち切り判定
            if self.prob_max <= self.log_post_prob:
                #更新する
                self.prob_max = self.log_post_prob.copy()
                self.s_hat = self.s.copy()
                self.c_hat = self.c
                self.mu_hat = self.mu.copy()
                self.Lambda_hat = self.Lambda.copy()
                self.n_hat = self.n_.copy()
                count = 0

            else:
                count +=1
                if count == stop_count:
                    break

            step += 1

    #計算に必要な関数の定義
    def calc_params(self, c_num, x, s, n_, mu0, beta, nu, S):
        '''
        各クラスタでパラメータを求める
        c_num:クラスタ番号を指定
        '''
        xs = np.c_[x, s] #パターンと潜在変数を結合したデータを作成

        #x_：クラスタに所属するデータ全体の平均を計算する
        x_ = np.mean(xs[xs[:,2]==c_num], axis=0)[:2]

        #最尤推定値:μ_cを計算する
        mu_c = (n_[c_num] * x_ + beta * mu0) / (n_[c_num] + beta)

        #Sqを計算する
        Sq_inv = np.zeros(S.shape)
        for xs_k in xs:
            if xs_k[2] == c_num:
                vector = (xs_k[:2] - x_)[:, np.newaxis]
                Sq_inv += np.dot(vector, vector.T)
        vector2 = (x_ - mu0)[:, np.newaxis]
        Sq_inv = np.linalg.inv(S) + ((n_[c_num] * beta)/(n_[c_num] + beta)) * np.dot(vector2, vector2.T)
        Sq = np.linalg.inv(Sq_inv)

        #ν_cを計算する
        nu_c = nu + n_[c_num]

        #Λ_iをウィシャート分布から生成する
        Lambda_i = st.wishart.rvs(df=nu_c, scale=Sq)

        #Λ_cを計算する
        Lambda_c = (n_[c_num] + beta) * Lambda_i

        #μ_iを計算する
        mu_i = st.multivariate_normal.rvs(mu_c, np.linalg.inv(Lambda_c))

        return mu_i, Lambda_i


    def calc_post_prob(self, x, s, c, n_, mu, Lambda, alpha, beta, n, mu0, nu, S):
        '''
        事後確率(対数)を計算する
        '''
        log_p = 0
        log_p += c * np.log(alpha)

        for i in range(c):
            for n in range(n_[i]-1):
                log_p += np.log(n+1)

            for x_k, s_k in zip(x, s):
                if s_k == i:
                    p_x = st.multivariate_normal.pdf(x_k, mu[i], np.linalg.inv(Lambda[i]))
                    log_p += np.log(p_x)

            #G_0(θ_i)を計算する
            p_N = st.multivariate_normal.pdf(mu[i], mu0, np.linalg.inv(beta*Lambda[i]))
            p_W = st.wishart.pdf(Lambda[i], df=nu, scale=S)
            G0_theta = p_N * p_W
            log_p += np.log(G0_theta)

        for i in range(n):
            log_p -= np.log(i+alpha)

        return log_p
