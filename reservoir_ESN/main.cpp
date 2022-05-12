#include <iostream>
#include <iomanip>
#include <vector>
#include <string>
#include <cmath>
#include <chrono>
#include <cblas.h>
#include <chrono>
#include <random>
#include <algorithm>
#include <cstring>
#include <execution>
#include <memory>
#include <numeric>
#include <map>
#include <algorithm>
#include <cmath>
#include "reservoir_perf.h"
#include "output_learning.h"
#include "task.h"
#include "tasks.h"
#include "output.h"

#define PHASE_NUM (3)
#define TRAIN (0)
#define VAL (1)
#define TEST (2)
#define MAX_UNIT_SIZE (200)
#define MAX_TASK_SIZE (1000)
#define THREAD_NUM (16)
#define SUBSET_SIZE (THREAD_NUM * 1)

#include <sstream>


typedef void (*FUNC)();
int main(void) {

	const int TRIAL_NUM = 2;	// ループ回数
	const int step = 4000;
	const int wash_out = 400;
	std::vector<int> unit_sizes = { 100, 100 };
	std::vector<std::string> toporogy = { "random", "ring" };
	std::vector<std::string> task_names = { "NL", "NL" };
	if (unit_sizes.size() != task_names.size()) return 0;

	const int lambda_step = 4;

	for (int r = 0; r < unit_sizes.size(); r++) {

		const int unit_size = unit_sizes[r];
		const std::string task_name = task_names[r];
		const std::string toporogy_type = toporogy[r];

		std::vector<std::vector<std::vector<std::vector<double>>>> output_node(SUBSET_SIZE, std::vector<std::vector<std::vector<double>>>(PHASE_NUM, std::vector<std::vector<double>>(step + 1, std::vector<double>(unit_size + 1, 0))));

		int connection_degree = unit_size / 10;
		if (toporogy_type == "ring") connection_degree = 1;
		std::vector<std::vector<double>> input_signal(PHASE_NUM);
		std::vector<std::vector<std::pair<std::string, std::vector<double>>>> teacher_signals(PHASE_NUM);

		std::vector<std::string> function_names = { "sinc", "tanh" };
		double alpha_min, d_alpha;
		double sigma_min, d_sigma;

		std::ofstream outputfile("output_data/" + task_name + std::to_string(unit_size) + "_" + toporogy_type + ".csv");
		
		// タスク(入力信号 教師信号)の生成
		tasks reservoir_task[PHASE_NUM] = { tasks(step, 0), tasks(step, 1), tasks(step, 2) };
		for (int phase = 0; phase < PHASE_NUM; phase++) {
			reservoir_task[phase].generate_random_input(-1.0, 1.0);
			reservoir_task[phase].generate_L_task(unit_size);
			reservoir_task[phase].generate_NL_task();
			reservoir_task[phase].generate_approx_task();
			reservoir_task[phase].generate_narma_task();
		}

		// 設定出力
		output_NL_format();

		std::cerr << reservoir_task[TRAIN].output_tasks.size() << std::endl;
		auto reservoir_set = reservoir_layer::generate_reservoir(unit_size, connection_degree, function_names, TRIAL_NUM, wash_out, toporogy_type);

		std::vector<reservoir_layer> reservoir_subset;

		std::chrono::system_clock::time_point  start, end;
		start = std::chrono::system_clock::now();

		// リザーバ集合を処理する
		for (int re = 0; re < reservoir_set.size(); re++) {

			// 高速化のため、まとめて(=SUBSET_SIZEごとに)リザーバを処理するようにする。
			reservoir_subset.push_back(reservoir_set[re]);
			if (reservoir_subset.size() < SUBSET_SIZE && re + 1 < reservoir_set.size()) {
				continue;
			}

			/*** 入力によるリザーバの時間発展 ***/
			std::cerr << "reservoir_update..." << std::endl;
#pragma omp parallel for num_threads(THREAD_NUM)
			for (int k = 0; k < reservoir_subset.size(); k++) {
				reservoir_subset[k].generate_reservoir();
				reservoir_subset[k].reservoir_update(reservoir_task[TRAIN].input_signal, output_node[k][TRAIN], step);
				reservoir_subset[k].reservoir_update(reservoir_task[VAL].input_signal, output_node[k][VAL], step);
				reservoir_subset[k].reservoir_update(reservoir_task[TEST].input_signal, output_node[k][TEST], step);
				reservoir_subset[k].calc_echo_state_property(reservoir_task[VAL].input_signal);
			}
			std::vector<reservoir_layer> reservoir_subset_tmp;
			for (int k = 0; k < reservoir_subset.size(); k++) {
				if (reservoir_subset[k].is_echo_state_property) {
					output_node[reservoir_subset_tmp.size()][TRAIN] = output_node[k][TRAIN];
					output_node[reservoir_subset_tmp.size()][VAL] = output_node[k][VAL];
					output_node[reservoir_subset_tmp.size()][TEST] = output_node[k][TEST];
					reservoir_subset_tmp.push_back(reservoir_subset[k]);
				}
			}
			reservoir_subset = reservoir_subset_tmp;
			int opt_k = 0;
			int i, t, j, lm;
			std::vector<output_learning> output_learning(reservoir_subset.size());
			std::vector<std::vector<double>> output_node_T(reservoir_subset.size());

			std::vector<std::vector<double>> opt_nmse(SUBSET_SIZE, std::vector<double>(reservoir_task[TRAIN].output_tasks.size(), 2));
			std::vector < std::vector <std::vector<double>>> opt_w(SUBSET_SIZE, std::vector <std::vector<double>>(reservoir_task[TRAIN].output_tasks.size()));
			std::cerr << "reservoir_training..." << std::endl;
#pragma omp parallel for num_threads(THREAD_NUM)
			// 重みの学習を行う
			for (int k = 0; k < reservoir_subset.size(); k++) {
				output_learning[k].generate_simultaneous_linear_equationsA(output_node[k][TRAIN], wash_out, step, unit_size);
				output_learning[k].generate_simultaneous_linear_equationsA_tilda(lambda_step, LAMBDA_MIN, LAMBDA_ADD);
			}
#pragma omp parallel for  private(lm, i, t, j) num_threads(THREAD_NUM)
			for (int k = 0; k < reservoir_subset.size(); k++) {
				for (i = 0; i <= unit_size; i++) {
					for (t = wash_out + 1; t < step; t++) {
						output_node_T[k].push_back(output_node[k][TRAIN][t + 1][i]);
					}
				}
			}
#pragma omp parallel for  private(lm, i, t, j) num_threads(THREAD_NUM)
			for (int k = 0; k < reservoir_subset.size(); k++) {
				output_learning[k].w.resize(reservoir_task[TRAIN].output_tasks.size(), std::vector<std::vector<double>>(lambda_step));
				output_learning[k].nmse.resize(reservoir_task[TRAIN].output_tasks.size(), std::vector<double>(lambda_step));
				for (lm = 0; lm < lambda_step; lm++) {
					output_learning[k].IncompleteCholeskyDecomp2(lm, unit_size + 1);
				}
				for (i = 0; i < reservoir_task[TRAIN].output_tasks.size(); i++) {
					output_learning[k].generate_simultaneous_linear_equationsb_fast(output_node_T[k], reservoir_task[TRAIN].output_tasks[i].output_signal, wash_out, step, unit_size);
					for (lm = 0; lm < lambda_step; lm++) {
						double eps = 1e-12;
						int itr = 10;
						output_learning[k].ICCGSolver(lm, output_learning[k].w[i][lm], unit_size + 1, itr, eps);
						output_learning[k].nmse[i][lm] = calc_nmse(reservoir_task[VAL].output_tasks[i].output_signal, output_learning[k].w[i][lm], output_node[k][VAL], unit_size, wash_out, step, false);
					}
				}
			}
			std::cerr << "reservoir_selecting..." << std::endl;

			// 検証データでもっとも性能の良いリザーバーを選択
			for (int k = 0; k < reservoir_subset.size(); k++) {
				for (int i = 0; i < reservoir_task[TRAIN].output_tasks.size(); i++) {
					for (int lm = 0; lm < lambda_step; lm++) {
						if (lm == 0 || output_learning[k].nmse[i][lm] < opt_nmse[k][i]) {
							opt_nmse[k][i] = output_learning[k].nmse[i][lm];
							opt_w[k][i] = output_learning[k].w[i][lm];
						}
					}
				}
			}

			/***  test_data による結果　***/

			std::vector<reservoir_perf> perf(reservoir_subset.size());
			std::cerr << "calc test_data nmse..." << std::endl;
			for (int k = 0; k < reservoir_subset.size(); k++) {
				std::cerr << k << ",";
				perf[k].reservoir = reservoir_subset[k];
				perf[k]._tasks = reservoir_task[TEST];

				// テストデータでnmseを計算する。
#pragma omp parallel for  private(i) num_threads(THREAD_NUM)
				for (i = 0; i < reservoir_task[TEST].output_tasks.size(); i++) {
					const double test_nmse = calc_nmse(reservoir_task[TEST].output_tasks[i].output_signal, opt_w[k][i], output_node[k][TEST], unit_size, wash_out, step, false);
					reservoir_task[TEST].output_tasks[i].nmse = test_nmse;
				}

				// タスクを追加する。
				perf[k].add_task(reservoir_task[TEST]);
			}
			std::cerr << "finished" << std::endl;


			/***  結果をファイルに出力　***/
			std::cerr << "output..." << std::endl;
			for (int k = 0; k < reservoir_subset.size(); k++) {
				perf[k].reservoir_perf_output(outputfile);
			}
			reservoir_subset.clear();


			// 進捗出力
			end = std::chrono::system_clock::now();  // 計測終了時間
			double elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count(); //処理に要した時間をミリ秒に変換
			std::cerr << (re + 1) << "/" << reservoir_set.size();
			std::cerr << std::fixed << std::setprecision(3);
			std::cerr << " [" << (re + 1.0) * 100 / reservoir_set.size() << "%] ";
			std::cerr << " elapsed time " << elapsed / 1000.0 / 3600.0 << "[hours]";
			std::cerr << " remain est." << (elapsed / 1000.0) / (re + 1) * (reservoir_set.size() - (re + 1)) / 3600.0 << "[hours]" << std::endl;

		}
		outputfile.close();
	}
}