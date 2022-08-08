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
parser.add_argument('--taskes',default=["approx_1_1.0", "approx_3_0.0", "approx_6_-0.5", "narma_5", "narma_10", "henon_3", "laser1_2", "count_4", "count_5", "laser0_2", "laser0_3", "laser1_3", "laser2_2", "laser2_3", "henon_4"], nargs='*')

args = parser.parse_args()
if not os.path.isdir(args.root + args.output):
    os.mkdir(args.root + args.output)
for path in args.data:
    for NL_type in args.NL_types:
        NL_max = 1000
        if NL_type == "NL_test":
            NL_max = 1000
        df = pd.read_csv(args.root + path + '.csv', sep=',',comment='#')
        
        for topology in ["random"]:
        # for topology in ["sparse_random"]:
            #df2 = df[df["topology"] == topology]
            df2 = df[df["L"] > 0.5]
            # df2 = df2[df2["feed_gain"] < 0.15]
            # print(df2["feed_gain"])
            for task in args.taskes:
                fig = plt.figure()
                ax1 = fig.add_subplot(1, 3, 1)
                ax2 = fig.add_subplot(1, 3, 2)
                ax3 = fig.add_subplot(1, 3, 3)
                for num, function_name, bias, ax in [(1, "all", True, ax1), (2, "TD_exp", True, ax2), (3, "TD_ikeda", True, ax3)]:
                    df3 = df2[df2["function_name"] == function_name]
                    # df3 = df2
                    # if bias:
                    #     df3 = df3[df2["bias_factor"] != 0]
                    # else:
                    #     df3 = df3[df2["bias_factor"] == 0]
                    #df3.to_csv(task + ".csv")
                    if function_name == "all":
                        df3 = df2
                    # data_x = df3["NL_old_2"] + df3["L_test"]
                    data_x = df3["L_test"]
                    if NL_type == "NL_test" and False:
                        data_y = df3["NL_old_2"] + df3["NL_old_3"] + df3["NL_old_4"] + df3["NL_old_5"] + df3["NL_old_6"]
                    else:
                        data_y = df3[NL_type]
                    
                    value = df3[task]
                    
                    # print(data_x)
                    norm = LogNorm(vmin=0.05, vmax=1.0)
                    if "henon" in task:
                        norm = LogNorm(vmin=0.03, vmax=1.0)
                    if "count" in task:
                        norm = LogNorm(vmin=0.03, vmax=1.0)
                    if "approx" in task:
                        norm = LogNorm(vmin=0.1, vmax=1.0)
                    for i in value.index:
                        if type(value[i]) is not str:
                            continue
                        if "nan" not in value[i]:
                            value[i] = float(value[i])
                        else:
                            value[i]  = 10.0
                    mappable = ax.scatter(data_x, data_y,s=1, marker=".", c = value, alpha=1, linewidths=1, norm=norm, cmap='viridis_r')
                    if num != 1:
                        ax.tick_params(axis = 'y', labelcolor = "None")
                    ax.set_ylim(0.0, NL_max)
                    ax.set_xlim(0.0, 100)
                    ax.set_title(function_name + "," + str(bias))
                    #plt.legend(loc = "best")
                print(args.root + args.output + path + "_" + task + "_" + NL_type + "_" + topology + ".png")
                fig.colorbar(mappable, ax=ax)
                fig.savefig(args.root + args.output + path + "_" + task + "_" + NL_type + "_" + topology + ".png", dpi = 600)
                fig.clf()
