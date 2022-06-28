import pandas as pd
import os
from matplotlib import pyplot as plt
from matplotlib.colors import LogNorm

#for path in ["NL_0_0.0_50_ring","NL_0_0.0_50_random","NL_0_0.0_100_ring","NL_0_0.0_100_random"]:
import argparse
parser = argparse.ArgumentParser(description='NL_L平面を描写する')
parser.add_argument('--root', default= "..//output_data//", help='ルートパスを指定する')
parser.add_argument('--output', default="fig//")
parser.add_argument('--data', default= ["NL100"], help='ファイルを指定する', nargs='*')
parser.add_argument('--NL_types',default = ["NL_test"], help = "NL_typeを指定する", nargs='*')
parser.add_argument('--taskes',default=["approx_3_0.0", "approx_6_-0.5", "approx_11_-1.0", "narma_5", "narma_10", "henon_3", "laser1_2", "count_5", "count_6", "laser0_2", "laser0_3", "laser1_3", "laser2_2", "laser2_3", "henon_4"], nargs='*')

args = parser.parse_args()
if not os.path.isdir(args.root + args.output):
    os.mkdir(args.root + args.output)
for path in args.data:
    for NL_type in args.NL_types:
        NL_max =400
        if NL_type == "NL_test":
            NL_max = 400
        df = pd.read_csv(args.root + path + '.csv', sep=',',comment='#')
        
        for topology in ["random", "ring", "sparse_random", "doubly_ring"]:
        # for topology in ["sparse_random"]:
            df2 = df[df["topology"] == topology]
       

            for task in args.taskes:
                fig = plt.figure()
                ax1 = fig.add_subplot(1, 5, 1)
                ax2 = fig.add_subplot(1, 5, 2)
                ax3 = fig.add_subplot(1, 5, 3)
                ax4 = fig.add_subplot(1, 5, 4)
                ax5 = fig.add_subplot(1, 5, 5)
                for num, function_name, bias, ax in [(1, "all", True, ax1), (2, "sinc", True, ax2), (3, "tanh", True, ax3), (4, "sinc", False, ax4),(5, "tanh", False, ax5)]:
                    df3 = df2[df2["function_name"] == function_name]

                    if bias:
                        df3 = df3[df2["p"] > 0.99]
                    else:
                        df3 = df3[df2["p"] < 0.97]
                    #df3.to_csv(task + ".csv")
                    if function_name == "all":
                        df3 = df2
                    print(df3)
                    data_x = df3["L_test"]
                    if NL_type == "NL_test":
                        #data_y = df3["NL_old_2"] + df3["NL_old_3"] + df3["NL_old_4"] + df3["NL_old_5"] + df3["NL_old_6"]
                        data_y = df3[NL_type]
                    else:
                        data_y = df3[NL_type]
                
                    value = df3[task]
                    norm = LogNorm(vmin=0.05, vmax=1.0)
                    if "henon" in task:
                        norm = LogNorm(vmin=0.03, vmax=1.0)
                    if "count" in task:
                        norm = LogNorm(vmin=0.03, vmax=1.0)
                    if "approx" in task:
                        norm = LogNorm(vmin=0.1, vmax=1.0)
                    mappable = ax.scatter(data_x, data_y,s=1, marker=".", c = value, alpha=1, linewidths=1, norm=norm, cmap='viridis_r')
                    if num != 1:
                        ax.tick_params(axis = 'y', labelcolor = "None")
                    ax.set_ylim(0, 300)
                    ax.set_xlim(0, 300)
                    ax.set_title(function_name + "," + str(bias))
                    #plt.legend(loc = "best")
                print(args.root + args.output + path + "_" + task + "_" + NL_type + "_" + topology + ".png")
                fig.colorbar(mappable, ax=ax)
                fig.savefig(args.root + args.output + path + "_" + task + "_" + NL_type + "_" + topology + ".png", dpi = 600)
                fig.clf()
