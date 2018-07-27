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

## ファイル概要
- data (ipynb上での確認に使用)
    - nikkei225_d.csv
    - usdjpy_d.csv
    - SF10.csv ()
    - kddcup.data.corrected
    - header.csv
    - simulation (数値計算の確認)
        - hellinger_simulate.xlsx
        - SDEM_simulate.xlsx
        - SDLE_simulate.xlsx
- src (作成プログラム)
    - em_2dim.py (2次元の入力に対応したEMアルゴリズム)
    - em.py (n次元の入力に対応したEMアルゴリズム)
    - md.py (マハラノビス距離クラス)
    - sdem.py (SDEMクラス)
    - sdle.py (SDLEクラス)
    - smartsifter.py (SmartSifterクラス)
- ipynb (プログラムの動作確認, 動作方法)
    - AccessTracer.ipynb
    - ChangeFinder.ipynb (ChangeFinderを日経平均に適用)
    - EM-algorithm(n-dim).ipynb (EMアルゴリズムの動作確認)
    - HMM.ipynb
    - HMM(Baum-Welch).ipynb
    - Mahalanobis_Distance.ipynb (マハラノビス距離の動作確認)
    - NaiveBayes.ipynb (ナイーブベイズ法の動作確認)
    - SF10.ipynb (SmartSigterをSF10.csvに適用)
    - SF10-test.ipynb (SF10.ipynbの実行が必須)
    - SmartSifter.ipynb (SmartSifterの動作確認)
    - SmartSifter_for_n225.ipynb (SmartSifterを日経平均に適用)
    - workspace.ipynb (様々な動作確認用)
    - tmp
        - EM-algorithm.ipynb
        - SmartSifter.ipynb
        - SmartSifter(class未実装).ipynb


