# coding: utf-8

import numpy as np
import numpy.random as rd
import pandas as pd
from scipy import stats as st

seed = 0
rd.seed(seed)

def CRP(alpha, N):
    '''
    中華料理店過程に基づいてクラスタ数未定のクラスタリングのための事前分布を決定する
    alpha:パラメータ(集中度), n:全体の人数
    '''

    s = [] #各一が座るテーブル番号のリスト
    table = {} #テーブルごとの人数の辞書

    for n in range(N):
        if n == 0:
            s.append(0) #一人目は無条件で0番目のテーブルに着席
            table.setdefault(0, 1)
            continue

        else:
            prob = rd.random() #0-1の範囲で乱数を取得
            prob_sum = 0 #各テーブルの着席確率を累積していき、probを超えたら着席

            #新規テーブルについて
            prob_n_table = alpha / (n - 1 + alpha)
            prob_sum += prob_n_table

            if prob_sum >= prob:
                s.append(len(table))
                table.setdefault(len(table), 1)
                continue

            #既存のテーブルについて
            for t in table.keys():
                prob_sit_table = table[t] / (n - 1 + alpha)
                prob_sum += prob_sit_table

                if prob_sum >= prob:
                    s.append(t)
                    table[t] += 1
                    break

    return s, table

if __name__ == '__main__':
    main()
