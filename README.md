# データマイニングによる異常検知

## 実行環境

<b>python == 3.6</b> <br>

numpy == 1.12.1<br>
scipy == 1.1.0<br>
pandas == 0.19.2<br>
matplotlib == 2.2.2<br>
seaborn == 0.7.1<br>
changefinder == 0.03<br>
jupyter == 1.0.0<br>

## 実装

- [x] 外れ値検出
    - [x] マハラノビス距離
    - [x] SmartSifter
- [x] 変化点検出
    - [ ] 統計的検定に基づく変化点検出
    - [x] ChangeFinder (ライブラリインストール)
    - [ ] ChangeFinder (2次元以上対応版)
- [x] 異常検知
    - [x] ナイーブベイズ法
    - [x] AccessTracer
- [ ] 潜在的異常検知
- [x] 中華料理店過程
    - [x] 生成過程
    - [x] クラスタリング 

## ファイル概要

- data (ipynb上での確認に使用)
    - nikkei225_d.csv
    - usdjpy_d.csv
    - SF10.csv
    - kddcup.data.corrected
    - header.csv
    - simulation (数値計算の確認)
        - hellinger_simulate.xlsx
        - SDEM_simulate.xlsx
        - SDLE_simulate.xlsx
- src (作成プログラム)
    - accesstracer.py (AccessTreserクラス)
    - baumwelch.py (Baum-Welchアルゴリズム)
    - crp.py (中華料理店過程)
    - dpm.py (ディリクレ過程によるクラスタリング)
    - em_2dim.py (2次元の入力に対応したEMアルゴリズム)
    - em.py (n次元の入力に対応したEMアルゴリズム)
    - le (レビンソン法クラス)
    - md.py (マハラノビス距離クラス)
    - sdem.py (SDEMクラス)
    - sdar.py (SDARクラス)
    - sdhm.py (SDHMクラス)
    - sdle.py (SDLEクラス)
    - smartsifter.py (SmartSifterクラス)
- ipynb (プログラムの動作確認, 動作方法)
    - AccessTracer.ipynb (AccessTracerを日経平均に適用)
    - ChangeFinder.ipynb (ChangeFinderを日経平均に適用)
    - DPM.ipynb (中華料理店過程の動作確認)
    - Mahalanobis_Distance.ipynb (マハラノビス距離の動作確認)
    - NaiveBayes.ipynb (ナイーブベイズ法の動作確認)
    - SF10.ipynb (SmartSigterをSF10.csvに適用)
    - SF10-test.ipynb (SF10.ipynbの実行が必須)
    - SmartSifter.ipynb (SmartSifterの動作確認)
    - unix_command_anomaly_detection (SmartSifterをunixコマンドデータに適用)
    - tmp
        - CRP.ipynb
        - DPM_memo.ipynb
        - DPM_memo.ipynb
        - EM-algorithm.ipynb
        - EM-algorithm(n-dim).ipynb (EMアルゴリズムの動作確認)
        - HMM.ipynb
        - HMM(Baum-Welch).ipynb
        - SmartSifter_class.ipynb
        - SmartSifter.ipynb
        - SmartSifter(class未実装).ipynb
        - SmartSifter_for_n225.ipynb (SmartSifterを日経平均に適用)
        - workspace.ipynb (様々な動作確認用)


