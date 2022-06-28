import pandas as pd
import os
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.colors import LogNorm

#for path in ["NL_0_0.0_50_ring","NL_0_0.0_50_random","NL_0_0.0_100_ring","NL_0_0.0_100_random"]:
import argparse
parser = argparse.ArgumentParser(description='NL_L平面を描写する')
parser.add_argument('--root', default= "..//output_data//", help='ルートパスを指定する')
parser.add_argument('--output', default="fig//")
parser.add_argument('--data', default= ["NL100"], help='ファイルを指定する', nargs='*')
parser.add_argument('--NL_types',default = ["NL_test"], help = "NL_typeを指定する", nargs='*')
parser.add_argument('--taskes',default=["approx_3_0.0", "approx_6_-0.5", "approx_11_-1.0", "narma_5", "narma_10", "henon_3", "laser1_2", "count_4", "count_5", "laser0_2", "laser0_3", "laser1_3", "laser2_2", "laser2_3", "henon_4", "henon_5"], nargs='*')

args = parser.parse_args()
for path in args.data:
    df = pd.read_csv(args.root + path + '.csv', sep=',',comment='#')
    df.describe().to_csv("statictics.csv")
    for task in args.taskes:
        df = df.sort_values(task)
        df_100 = df.head(500)
        for topology in df_100["topology"].unique():
            count = (df_100["topology"] == topology).sum()
            average = df_100[df_100["topology"] == topology][task].min()
            print(task, topology, count/500.0, average)
        for p in np.sort(df_100["p"].unique()):
            count = (df_100["p"] == p).sum()
            average = df_100[df_100["p"] == p][task].min()
            print(task, p, count/500.0, average)
