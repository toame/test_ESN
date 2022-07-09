from re import I
import pandas as pd
import os
from matplotlib import pyplot as plt
from matplotlib.colors import LogNorm

#for path in ["NL_0_0.0_50_ring","NL_0_0.0_50_random","NL_0_0.0_100_ring","NL_0_0.0_100_random"]:
import argparse
parser = argparse.ArgumentParser(description='NL_L平面を描写する')
parser.add_argument('--root', default= "..//output_data//", help='ルートパスを指定する')
parser.add_argument('--output', default="fig//")
parser.add_argument('--data', default= ["NL_0_0.0_100_random"], help='ファイルを指定する', nargs='*')
parser.add_argument('--NL_types',default = ["NL_old"], help = "NL_typeを指定する", nargs='*')
parser.add_argument('--taskes',default=["approx_2_-1.0", "approx_4_-1.5", "approx_6_-2.0", "narma_5", "narma_10", "henon_3", "laser1_2", "count_4", "count_5", "laser0_2", "laser0_3", "laser1_3", "laser2_2", "laser2_3", "henon_4"], nargs='*')

args = parser.parse_args()
if not os.path.isdir(args.root + args.output):
    os.mkdir(args.root + args.output)
for path in args.data:
    for NL_type in args.NL_types:
        NL_max = 1000
        if NL_type == "NL_test":
            NL_max = 1000
        df = pd.read_csv(args.root + path + '.csv', sep=',',comment='#')
        df = df[df["NL_old_2"] + df['L'] >= 40]
        print(df)
        df.to_csv("test.csv")
       