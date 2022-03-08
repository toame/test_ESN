﻿#include <iostream>
#include <iomanip>
#include <vector>
#include <string>
#include <cmath>
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
#include <cblas.h>
#include "reservoir_layer.h"
#include "output_learning.h"
#include "task.h"
#include "tasks.h"
#define PHASE_NUM (3)
#define TRAIN (0)
#define VAL (1)
#define TEST (2)
#define MAX_UNIT_SIZE (200)
#define MAX_TASK_SIZE (1000)
#define TRUNC_EPSILON (1.7e-4)
#define THREAD_NUM (16)
#define SUBSET_SIZE (THREAD_NUM * 1)

#include <sstream>

template <typename T>
std::string to_string_with_precision(const T a_value, const int n = 6)
{
	std::ostringstream out;
	out.precision(n);
	out << std::fixed << a_value;
	return out.str();
}

typedef void (*FUNC)();
int main(void) {

	const int TRIAL_NUM = 2;	// ループ回数
	const int step = 4000;
	const int wash_out = 400;
	std::vector<int> unit_sizes = { 50, 50 };
	std::vector<std::string> toporogy = { "random", "ring" };
	std::vector<std::string> task_names = { "NL", "NL" };
	if (unit_sizes.size() != task_names.size()) return 0;
	std::vector<int> param1 = { 0, 0 };
	std::vector<double> param2 = { 0, 0 };
	if (param1.size() != param2.size()) return 0;


	std::vector<double> p_set{ 0.05, 0.1, 0.2, 0.35, 0.5, 0.65, 0.8, 0.9, 0.95, 1.0, 0.0 };
	std::vector<double> bias_set{ 0, 1, 2, 3, 5, 8 };
	std::vector<double> alpha_set{ 0.01, 0.02, 0.03, 0.04, 0.06, 0.08, 0.1, 0.15, 0.2, 0.3, 0.4, 0.6, 0.8, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0, 6.0, 8.0, 10.0 };
	std::vector<double> sigma_set{ 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2 };

	//std::vector<double> p_set{ 0.0, 0.5, 1.0 };
	//std::vector<double> bias_set{ 0, 1, 2, 5};
	//std::vector<double> alpha_set{ 0.01, 0.03, 0.1, 0.3, 0.6, 1.0, 3.0, 6.0, 10.0 };
	//std::vector<double> sigma_set{ 0.1, 0.25, 0.4, 0.6, 0.8, 0.9, 1.0};

	const int alpha_step = alpha_set.size();
	const int sigma_step = sigma_set.size();
	const int lambda_step = 4;
	std::string task_name;
	std::string function_name;


	std::vector<double> approx_nu_set({ -3.0, -2.5, -2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0 });
	std::vector<double> approx_tau_set({ -2, 0, 1, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0 });
	std::vector<int> narma_tau_set({ 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20 });

	std::ofstream output_NL_d("output_data/d_vec.csv");

	std::vector<std::vector<std::vector<int>>> d_vec(PHASE_NUM);
	generate_d_sequence_set(d_vec);
	for (auto& d : d_vec[TRAIN]) {
		for (int i = 0; i < d.size(); i++) {
			if (i != 0) output_NL_d << ",";
			output_NL_d << d[i];
		}
		output_NL_d << std::endl;
	}
	output_NL_d.close();

	for (int r = 0; r < unit_sizes.size(); r++) {

		const int unit_size = unit_sizes[r];
		const std::string task_name = task_names[r];
		const std::string toporogy_type = toporogy[r];

		std::vector<std::vector<std::vector<std::vector<double>>>> output_node(SUBSET_SIZE, std::vector<std::vector<std::vector<double>>>(PHASE_NUM, std::vector<std::vector<double>>(step + 1, std::vector<double>(unit_size + 1, 0))));
		std::vector < std::vector<std::vector<std::vector<double>>>> w(SUBSET_SIZE, std::vector<std::vector<std::vector<double>>>(MAX_TASK_SIZE, std::vector<std::vector<double>>(lambda_step))); // 各リザーバーの出力重み
		std::vector<std::vector<std::vector<double>>> nmse(SUBSET_SIZE, std::vector<std::vector<double>>(MAX_TASK_SIZE, std::vector<double>(lambda_step)));						// 各リザーバーのnmseを格納

		int connection_degree = unit_size / 10;
		if (toporogy_type == "ring") connection_degree = 1;
		std::vector<std::vector<double>> input_signal(PHASE_NUM);
		std::vector<std::vector<std::pair<std::string, std::vector<double>>>> teacher_signals(PHASE_NUM);

		std::vector<std::string> function_names = { "sinc", "tanh" };
		double alpha_min, d_alpha;
		double sigma_min, d_sigma;

		std::ofstream outputfile("output_data/" + task_name + "_" + std::to_string(param1[r]) + "_" + to_string_with_precision(param2[r], 1) + "_" + std::to_string(unit_size) + "_" + toporogy_type + ".csv");
		
		// 入力信号 教師信号の生成
		tasks reservoir_task[PHASE_NUM] = { tasks(step, 0), tasks(step, 1), tasks(step, 2) };
		for (int phase = 0; phase < PHASE_NUM; phase++) {
			reservoir_task[phase].generate_random_input(-1.0, 1.0);
			reservoir_task[phase].generate_L_task(unit_size);
			reservoir_task[phase].generate_NL_task();
			reservoir_task[phase].generate_approx_task(approx_tau_set, approx_nu_set);
			reservoir_task[phase].generate_narma_task(narma_tau_set);
			
		}

		// 設定出力
		outputfile << "topology,function_name,seed,unit_size,p,input_signal_factor,bias_factor,weight_factor,L,L_cut,NL,NL_old,NL1_old,NL_old_cut1,NL_old_cut2";
		for (int i = 2; i <= 7; i++) outputfile << ",NL_old_" << std::to_string(i);
		for (int i = 0; i < reservoir_task[TRAIN].output_tasks.size(); i++) {
			outputfile << "," << reservoir_task[TRAIN].output_tasks[i].task_name;
		}
		outputfile << std::endl;
		std::cerr << reservoir_task[TRAIN].output_tasks.size() << std::endl;
		auto reservoir_set = reservoir_layer::generate_reservoir(p_set, bias_set, alpha_set, sigma_set, unit_size, connection_degree, function_names, 2, wash_out, toporogy_type);

		std::vector<reservoir_layer> reservoir_subset;
		std::chrono::system_clock::time_point  start, end; // 型は auto で可
		start = std::chrono::system_clock::now(); // 計測開始時間
	   // 処理
		for (int re = 0; re < reservoir_set.size(); re++) {
			reservoir_subset.push_back(reservoir_set[re]);
			if (reservoir_subset.size() < SUBSET_SIZE && re + 1 < reservoir_set.size()) {
				continue;
			}
			

			std::vector<std::vector<double>> opt_nmse(SUBSET_SIZE, std::vector<double>(reservoir_task[TRAIN].output_tasks.size(), 2));
			std::vector < std::vector<double>> opt_lm2(SUBSET_SIZE, std::vector<double>(reservoir_task[TRAIN].output_tasks.size()));
			std::vector < std::vector <std::vector<double>>> opt_w(SUBSET_SIZE, std::vector <std::vector<double>>(reservoir_task[TRAIN].output_tasks.size()));

			// 複数のリザーバーの時間発展をまとめて処理
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
			std::cerr << reservoir_subset.size() << "," << reservoir_subset_tmp.size() << std::endl;
			reservoir_subset = reservoir_subset_tmp;
			int lm;

			int opt_k = 0;
			int i, t, j;
			std::vector<output_learning> output_learning(1000);
			std::vector<std::vector<double>> A(1000);
			std::vector<std::vector<double>> output_node_T(1000);
			std::cerr << "reservoir_training..." << std::endl;
#pragma omp parallel for num_threads(THREAD_NUM)
			// 重みの学習を行う
			for (int k = 0; k < reservoir_subset.size(); k++) {
				output_learning[k].generate_simultaneous_linear_equationsA(output_node[k][TRAIN], wash_out, step, unit_size);
			}
#pragma omp parallel for private(i) num_threads(THREAD_NUM)
			for (int k = 0; k < reservoir_subset.size(); k++) {
				A[k].resize(unit_size + 1);
				for (i = 0; i <= unit_size; i++) {
					A[k][i] = output_learning[k].A[i][i];
				}
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
				for (lm = 0; lm < lambda_step; lm++) {
					//std::cerr << k << "," << lm << std::endl;
					for (j = 0; j <= unit_size; j++) {
						output_learning[k].A[j][j] = A[k][j] + pow(10, -14 + lm * 2);
					}
					output_learning[k].IncompleteCholeskyDecomp2(unit_size + 1);
					for (i = 0; i < reservoir_task[TRAIN].output_tasks.size(); i++) {
						output_learning[k].generate_simultaneous_linear_equationsb_fast(output_node_T[k], reservoir_task[TRAIN].output_tasks[i].output_signal, wash_out, step, unit_size);
						double eps = 1e-12;
						int itr = 10;
						output_learning[k].ICCGSolver(unit_size + 1, itr, eps);
						w[k][i][lm] = output_learning[k].w;
						nmse[k][i][lm] = calc_nmse(reservoir_task[VAL].output_tasks[i].output_signal, output_learning[k].w, output_node[k][VAL], unit_size, wash_out, step, false);
					}
				}
			}
			std::cerr << "reservoir_selecting..." << std::endl;

			// 検証データでもっとも性能の良いリザーバーを選択
			for (int k = 0; k < reservoir_subset.size(); k++) {
				for (int i = 0; i < reservoir_task[TRAIN].output_tasks.size(); i++) {
					for (int lm = 0; lm < lambda_step; lm++) {
						if (lm == 0 || nmse[k][i][lm] < opt_nmse[k][i]) {
							opt_nmse[k][i] = nmse[k][i][lm];
							opt_lm2[k][i] = lm;
							opt_w[k][i] = w[k][i][lm];
						}
					}
				}
			}

			std::vector<double> L(reservoir_subset.size());
			std::vector<double> NL(reservoir_subset.size());
			std::vector<double> NL_old(reservoir_subset.size());
			std::vector<double> NL1_old(reservoir_subset.size());
			std::vector<double> L_cut(reservoir_subset.size());
			std::vector<double> NL_old_cut1(reservoir_subset.size());
			std::vector<double> NL_old_cut2(reservoir_subset.size());
			std::vector <std::map<int, double>> sub_NL_old(reservoir_subset.size());
			std::vector<std::vector<double>> sub_L(reservoir_subset.size());
			std::vector<std::vector<double>> sub_NL(reservoir_subset.size());
			std::vector<std::map<int, double>> sub_NL_old2(reservoir_subset.size());
			std::vector<std::vector<double>> narma_task(reservoir_subset.size());
			std::vector<std::vector<double>> approx_task(reservoir_subset.size());
			std::vector<int> maxL(reservoir_subset.size());
			std::cerr << "calc_L, calc_NL..." << std::endl;
#pragma omp parallel for  private(i) num_threads(THREAD_NUM)
			for (int k = 0; k < reservoir_subset.size(); k++) {
				std::cerr << k << ",";
				for (i = 0; i < reservoir_task[TEST].output_tasks.size(); i++) {
					const double test_nmse = calc_nmse(reservoir_task[TEST].output_tasks[i].output_signal, opt_w[k][i], output_node[k][TEST], unit_size, wash_out, step, false);
					if (reservoir_task[TEST].output_tasks[i].task_label == "narma") narma_task[k].push_back(test_nmse);
					else if (reservoir_task[TEST].output_tasks[i].task_label == "approx") approx_task[k].push_back(test_nmse);
					else if (reservoir_task[TEST].output_tasks[i].task_label == "L") {
						int tau = stoi(reservoir_task[TEST].output_tasks[i].task_name.substr(2));
						const double tmp_L = 1.0 - test_nmse;
						if (tmp_L >= TRUNC_EPSILON) {
							L[k] += tmp_L;
							L_cut[k] += tmp_L * tmp_L;
						}
						if (tmp_L >= 0.9) {
							maxL[k] = std::max(tau, maxL[k]);
						}
						sub_L[k].push_back(std::max(0.0, tmp_L));
					}
					else if (reservoir_task[TEST].output_tasks[i].task_label == "NL2") {
						const double tmp_NL = (1.0 - test_nmse);
						if (tmp_NL >= TRUNC_EPSILON) NL[k] += tmp_NL;
						sub_NL[k].push_back(std::max(0.0, tmp_NL));
					}
					else if (reservoir_task[TEST].output_tasks[i].task_label == "NL") {
						int idx = stoi(reservoir_task[TEST].output_tasks[i].task_name.substr(3));
						int d_sum = 0;
						std::vector<int> d = d_vec[TEST][idx];
						int last = 0;
						for (int r = 0; r < d.size(); r++) {
							d_sum += d[r];
							if (d[r] > 0) last = r;
						}
						const double tmp_NL = d_sum * (1.0 - test_nmse);
						const double tmp_NL1 = (1.0 - test_nmse);
						sub_NL_old2[k][idx] = tmp_NL1;
						if (tmp_NL >= TRUNC_EPSILON) {
							NL1_old[k] += tmp_NL1;
							NL_old[k] += tmp_NL;
							sub_NL_old[k][d_sum] += tmp_NL1 * tmp_NL1 * d_sum;
							NL_old_cut1[k] += tmp_NL1 * tmp_NL1 * d_sum;
							if (last <= maxL[k]) {
								NL_old_cut2[k] += tmp_NL;
							}
						}

						
					}
					else {
						std::cerr << "error" << std::endl;
					}
				}

			}
			std::cerr << "output..." << std::endl;
			// ファイル出力
			for (int k = 0; k < reservoir_subset.size(); k++) {
				const double input_signal_factor = reservoir_subset[k].input_signal_factor;
				const double weight_factor = reservoir_subset[k].weight_factor;
				const int seed = reservoir_subset[k].seed;
				double bias_factor1 = reservoir_subset[k].bias_factor;
				if (bias_factor1 < 0) bias_factor1 = input_signal_factor * weight_factor;
				const double p = reservoir_subset[k].p;
				std::string function_name = reservoir_subset[k].nonlinear_name;
				outputfile << toporogy_type << "," << function_name << "," << seed << "," << unit_size << "," << p << "," << input_signal_factor << "," << bias_factor1 << "," << weight_factor;
				outputfile << "," << L[k] << "," << L_cut[k] << "," << NL[k] << "," << NL_old[k] << "," << NL1_old[k] << "," << NL_old_cut1[k] << "," << NL_old_cut2[k];

				for (int i = 2; i < 8; i++) outputfile << "," << sub_NL_old[k][i];
				for (int i = 0; i < std::min<int>(100, sub_L[k].size()); i++) outputfile << "," << sub_L[k][i];
				for (int i = 0; i < sub_NL_old2[k].size(); i++) outputfile << "," << sub_NL_old2[k][i];
				for (int i = 0; i < narma_task[k].size(); i++) outputfile << "," << narma_task[k][i];
				for (int i = 0; i < approx_task[k].size(); i++)	outputfile << "," << approx_task[k][i];
				outputfile << std::endl;
			}
			reservoir_subset.clear();
			end = std::chrono::system_clock::now();  // 計測終了時間
			double elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count(); //処理に要した時間をミリ秒に変換
			std::cerr << (re + 1) << "/" << reservoir_set.size() << " remain est." << (elapsed / 1000.0) / (re + 1) * (reservoir_set.size() - (re + 1)) / 3600.0 << "[hours]" << std::endl;

		}
		outputfile.close();
	}
}