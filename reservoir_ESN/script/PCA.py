import pandas as pd
import os
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.colors import LogNorm
import sklearn #機械学習のライブラリ
from sklearn.decomposition import PCA #主成分分析器
#for path in ["NL_0_0.0_50_ring","NL_0_0.0_50_random","NL_0_0.0_100_ring","NL_0_0.0_100_random"]:
import argparse
parser = argparse.ArgumentParser(description='NL_L平面を描写する')
parser.add_argument('--root', default= "..//output_data//", help='ルートパスを指定する')
parser.add_argument('--output', default="fig//")
parser.add_argument('--data', default= ["NL100_1"], help='ファイルを指定する', nargs='*')
parser.add_argument('--NL_types',default = ["NL_test"], help = "NL_typeを指定する", nargs='*')
parser.add_argument('--taskes',default=["approx_3_0.0", "approx_6_-0.5", "approx_11_-1.0", "narma_5", "narma_10", "henon_3", "laser1_2", "count_4", "count_5", "laser0_2", "laser0_3", "laser1_3", "laser2_2", "laser2_3", "henon_4", "henon_5"], nargs='*')

args = parser.parse_args()
for path in args.data:
    df = pd.read_csv(args.root + path + '.csv', sep=',',comment='#')
    df = df.iloc[:50000,:]
    df1 = df.iloc[:, 30:662]
    print(df1)
    dfs = df1.iloc[:, :].apply(lambda x: x, axis=0)
    #主成分分析の実行
    pca = PCA()
    pca.fit(dfs)
    # データを主成分空間に写像
    feature = pca.transform(dfs)
    task_name = "henon_3"
    SIZE = 4
    print(feature[:, :SIZE])
    pd.DataFrame(feature[:, :SIZE], columns=["PC{}".format(x + 1) for x in range(SIZE)]).head()
    plt.figure(figsize=(6, 6))
    # plt.scatter(feature[:, 0], feature[:, 1], s=3,alpha=1, c=list(df.loc[:, task_name]), vmin=0.03, vmax=1.0, cmap='viridis_r')
    plt.scatter(feature[:, 0], feature[:, 2], s=3,alpha=1, c=list(df.loc[:, task_name]),  vmin=0.05, vmax=1.0, cmap='viridis_r')
    plt.grid()
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.show()
    from pandas import plotting 
    plotting.scatter_matrix(pd.DataFrame(feature[:, :SIZE],
                        columns=["PC{}".format(x + 1) for x in range(SIZE)]), 
                        figsize=(8, 8), c=list(df.loc[:, task_name]), alpha=1, s = 3,  vmin=0.05, vmax=1.0, cmap='viridis_r') 
    plt.show()
    pd.DataFrame(pca.explained_variance_ratio_, index=["PC{}".format(x + 1) for x in range(len(dfs.columns))])
    import matplotlib.ticker as ticker
    plt.gca().get_xaxis().set_major_locator(ticker.MaxNLocator(integer=True))
    plt.plot([0] + list( np.cumsum(pca.explained_variance_ratio_)), "-o")
    plt.xlabel("Number of principal components")
    plt.ylabel("Cumulative contribution rate")
    plt.grid()
    plt.show()

    plt.figure(figsize=(6, 6))
    for x, y, name in zip(pca.components_[0], pca.components_[2], df1.columns[:]):
        plt.text(x, y, name)
        print(name, x, y)
    plt.scatter(pca.components_[0], pca.components_[2], alpha=0.8)
    plt.grid()
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.show()


