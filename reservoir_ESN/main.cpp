#include <iostream>
#include <iomanip>
#include <vector>
#include <string>
#include <cmath>
#include <cblas.h>
#include <chrono>
#include <random>
#include <algorithm>
#include <cstring>
#include <execution>
#include <memory>
#include <numeric>
#include <map>
#include "reservoir_layer.h"
#include "output_learning.h"
#include "task.h"
#define PHASE_NUM (3)
#define TRAIN (0)
#define VAL (1)
#define TEST (2)
#define MAX_UNIT_SIZE (200)
#define MAX_TASK_SIZE (3000)
#define TRUNC_EPSILON (1.7e-4)
#define THREAD_NUM (20)
#define SUBSET_SIZE (THREAD_NUM * 4)
double sinc(const double x) {
	if (x == 0) return 1.0;
	return sin(PI * x) / (PI * x);
}
double gauss(double y) { return exp(-y * y / (2.0 * 0.4 * 0.4)) / (sqrt(PI * 2) * 0.4); }
double oddsinc(double y) {
	if (y <= 0) return sin(PI * y) / (PI * (y + 1));
	else return sin(PI * -y) / (PI * (y - 1));
}
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

	std::vector<std::vector<std::vector<std::vector<double>>>> output_node(SUBSET_SIZE, std::vector<std::vector<std::vector<double>>>(PHASE_NUM, std::vector<std::vector<double>>(step + 2, std::vector<double>(MAX_UNIT_SIZE + 1, 0))));
	std::vector < std::vector<std::vector<std::vector<double>>>> w(SUBSET_SIZE, std::vector<std::vector<std::vector<double>>>(MAX_TASK_SIZE, std::vector<std::vector<double>>(lambda_step))); // 各リザーバーの出力重み
	std::vector<std::vector<std::vector<double>>> nmse(SUBSET_SIZE, std::vector<std::vector<double>>(MAX_TASK_SIZE, std::vector<double>(lambda_step)));						// 各リザーバーのnmseを格納
	std::vector<double> approx_nu_set({ -3.0, -2.5, -2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0, 6.5, 7.0 });
	std::vector<double> approx_tau_set({ -2, 0, 1, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0 });
	std::vector<int> narma_tau_set({ 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20 });
	for (int r = 0; r < unit_sizes.size(); r++) {
		const int unit_size = unit_sizes[r];
		const std::string task_name = task_names[r];
		const std::string toporogy_type = toporogy[r];
		int connection_degree = unit_size / 10;
		if (toporogy_type == "ring") connection_degree = 1;
		std::vector<std::vector<double>> input_signal(PHASE_NUM);
		std::vector<std::vector<std::vector<double>>> teacher_signals(PHASE_NUM);

		std::vector<std::string> function_names = { "sinc", "tanh" };
		double alpha_min, d_alpha;
		double sigma_min, d_sigma;

		std::ofstream outputfile("output_data/" + task_name + "_" + std::to_string(param1[r]) + "_" + to_string_with_precision(param2[r], 1) + "_" + std::to_string(unit_size) + "_" + toporogy_type + ".csv");
		// 入力信号 教師信号の生成
		std::vector<std::vector<std::vector<int>>> d_vec(PHASE_NUM);
		std::vector<std::string> task_name2;
		std::vector<int> task_size(5);
		generate_d_sequence_set(d_vec);
		for (int phase = 0; phase < PHASE_NUM; phase++) {
			generate_input_signal_random(input_signal[phase], -1.0, 2.0, step, phase + 1);
			for (auto tau : narma_tau_set) {
				std::vector<double> tmp_teacher_signal;
				generate_narma_task2(input_signal[phase], tmp_teacher_signal, tau, step);
				teacher_signals[phase].push_back(tmp_teacher_signal);
				if (phase == TRAIN) task_name2.push_back("narma" + std::to_string(tau));
			}
			task_size[0] = teacher_signals[phase].size();
			for (int i = 0; i < approx_nu_set.size(); i++) {
				for (int j = 0; j < approx_tau_set.size(); j++) {
					double nu = approx_nu_set[i];
					double tau = approx_tau_set[j];
					std::vector<double> tmp_teacher_signal;
					task_for_function_approximation2(input_signal[phase], tmp_teacher_signal, pow(2, nu), (int)(pow(2, tau) + 0.5), step, phase);
					teacher_signals[phase].push_back(tmp_teacher_signal);
					if (phase == TRAIN) task_name2.push_back("approx_" + std::to_string((int)(pow(2, tau) + 0.5)) + "_" + to_string_with_precision(nu, 1));
				}
			}
			task_size[1] = teacher_signals[phase].size();
			for (int tau = 1; tau <= unit_size; tau++) {
				std::vector<double> tmp_teacher_signal;
				task_for_calc_of_L(input_signal[phase], tmp_teacher_signal, tau, step);
				teacher_signals[phase].push_back(tmp_teacher_signal);
			}
			task_size[2] = teacher_signals[phase].size();
			std::cerr << "ok: " << d_vec[phase].size() << std::endl;
			for (int nu = 2; nu <= unit_size; nu++) {
				std::vector<double> tmp_teacher_signal;
				task_for_calc_of_NL2(input_signal[phase], tmp_teacher_signal, nu, step);
				teacher_signals[phase].push_back(tmp_teacher_signal);
			}
			task_size[3] = teacher_signals[phase].size();
			for (auto& d : d_vec[phase]) {
				std::vector<double> tmp_teacher_signal;
				task_for_calc_of_NL(input_signal[phase], tmp_teacher_signal, d, step);
				teacher_signals[phase].push_back(tmp_teacher_signal);
			}
			task_size[4] = teacher_signals[phase].size();
		}

		// 設定出力
		outputfile << "topology,function_name,seed,unit_size,p,input_signal_factor,bias_factor,weight_factor,L,NL,NL_old,NL1_old";
		for (int i = 2; i <= 7; i++) outputfile << ",NL_old_" << std::to_string(i);
		for (int i = 2; i <= 50; i++) outputfile << ",NL" << std::to_string(i);
		for (int i = 1; i <= 50; i++) outputfile << ",L" << std::to_string(i);
		for (int i = 55; i <= std::min(unit_size, 100); i += 5) outputfile << ",L" << std::to_string(i);
		for (int i = 0; i < task_name2.size(); i++) {
			outputfile << "," << task_name2[i];
		}
		outputfile << std::endl;
		std::cerr << teacher_signals[TRAIN].size() << std::endl;
		for (int i = 0; i < 4; i++) std::cerr << task_size[i] << " ";
		std::cerr << std::endl;
		std::chrono::system_clock::time_point  start, end; // 型は auto で可
		for (int loop = 0; loop < TRIAL_NUM; loop++) {
			start = std::chrono::system_clock::now(); // 計測開始時間
			for (auto function_name : function_names) {
				double (*nonlinear)(double);
				if (function_name == "sinc") nonlinear = sinc;
				else if (function_name == "tanh") nonlinear = tanh;
				else if (function_name == "gauss") nonlinear = gauss;
				else if (function_name == "oddsinc") nonlinear = oddsinc;
				else {
					std::cerr << "error! " << function_name << "is not found" << std::endl;
					return 0;
				}
				auto reservoir_set = reservoir_layer::generate_reservoir(p_set, bias_set, alpha_set, sigma_set, unit_size, connection_degree, nonlinear, loop, wash_out, toporogy_type);
				std::vector<reservoir_layer> reservoir_subset;
				for (int re = 0; re < reservoir_set.size(); re++) {
					reservoir_subset.push_back(reservoir_set[re]);
					if (reservoir_subset.size() < SUBSET_SIZE && re + 1 < reservoir_set.size()) {
						continue;
					}
					std::cerr << double((re + 1) * 100)/reservoir_set.size() << "[%]," <<  re << "," << reservoir_subset.size() << std::endl;

					std::vector<std::vector<double>> opt_nmse(SUBSET_SIZE, std::vector<double>(teacher_signals[TRAIN].size(), 1e+10));
					std::vector < std::vector<double>> opt_lm2(SUBSET_SIZE, std::vector<double>(teacher_signals[TRAIN].size()));
					std::vector < std::vector <std::vector<double>>> opt_w(SUBSET_SIZE, std::vector <std::vector<double>>(teacher_signals[TRAIN].size()));

					// 複数のリザーバーの時間発展をまとめて処理
					std::cerr << "reservoir_update..." << std::endl;
#pragma omp parallel for num_threads(THREAD_NUM)
					for (int k = 0; k < reservoir_subset.size(); k++) {
						reservoir_subset[k].generate_reservoir();

						reservoir_subset[k].reservoir_update(input_signal[TRAIN], output_node[k][TRAIN], step);
						reservoir_subset[k].reservoir_update(input_signal[VAL], output_node[k][VAL], step);
						reservoir_subset[k].reservoir_update(input_signal[TEST], output_node[k][TEST], step);
						reservoir_subset[k].judge_echo_state_property(input_signal[VAL]);
					}
					int lm;

					int opt_k = 0;
					int i, t, j;
					std::vector<output_learning> output_learning(1000);
					std::vector<std::vector<double>> A(1000);
					std::cerr << "reservoir_training..." << std::endl;
#pragma omp parallel for num_threads(THREAD_NUM)
					// 重みの学習を行う
					for (int k = 0; k < reservoir_subset.size(); k++) {
						if (!reservoir_subset[k].is_echo_state_property) continue;
						output_learning[k].generate_simultaneous_linear_equationsA(output_node[k][TRAIN], wash_out, step, unit_size);
					}
#pragma omp parallel for private(i) num_threads(THREAD_NUM)
					for (int k = 0; k < reservoir_subset.size(); k++) {
						if (!reservoir_subset[k].is_echo_state_property) continue;
						A[k].resize(unit_size + 1);
						for (i = 0; i <= unit_size; i++) {
							A[k][i] = output_learning[k].A[i][i];
						}
					}
#pragma omp parallel for  private(lm, i, t, j) num_threads(THREAD_NUM)
					for (int k = 0; k < reservoir_subset.size(); k++) {
						if (!reservoir_subset[k].is_echo_state_property) continue;
						std::vector<double> output_node_T;
						for (i = 0; i <= unit_size; i++) {
							for (t = wash_out + 1; t < step; t++) {
								output_node_T.push_back(output_node[k][TRAIN][t + 1][i]);
							}
						}
						for (i = 0; i < teacher_signals[TRAIN].size(); i++) {
							output_learning[k].generate_simultaneous_linear_equationsb_fast(output_node_T, teacher_signals[TRAIN][i], wash_out, step, unit_size);
							double opt_lm = 0;
							double opt_lm_nmse = 1e+9;
							for (lm = 0; lm < lambda_step; lm++) {
								for (j = 0; j <= unit_size; j++) {
									output_learning[k].A[j][j] = A[k][j] + pow(10, -14 + lm * 2);
								}
								output_learning[k].IncompleteCholeskyDecomp2(unit_size + 1);
								double eps = 1e-12;
								int itr = 10;
								output_learning[k].ICCGSolver(unit_size + 1, itr, eps);
								w[k][i][lm] = output_learning[k].w;
								nmse[k][i][lm] = calc_nmse(teacher_signals[VAL][i], output_learning[k].w, output_node[k][VAL], unit_size, wash_out, step, false);
							}
						}
					}
					std::cerr << "reservoir_selecting..." << std::endl;

					// 検証データでもっとも性能の良いリザーバーを選択
					for (int k = 0; k < reservoir_subset.size(); k++) {
						if (!reservoir_subset[k].is_echo_state_property) continue;
						for (int i = 0; i < teacher_signals[TRAIN].size(); i++) {
							for (int lm = 0; lm < lambda_step; lm++) {
								if (nmse[k][i][lm] < opt_nmse[k][i]) {
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
					std::vector<std::vector<double>> sub_NL_old(reservoir_subset.size());
					std::vector<std::vector<double>> sub_L(reservoir_subset.size());
					std::vector<std::vector<double>> sub_NL(reservoir_subset.size());
					std::vector<std::vector<double>> narma_task(reservoir_subset.size());
					std::vector<std::vector<double>> approx_task(reservoir_subset.size());
					std::cerr << "calc_L, calc_NL..." << std::endl;
#pragma omp parallel for  private(i) num_threads(THREAD_NUM)
					for (int k = 0; k < reservoir_subset.size(); k++) {
						if (!reservoir_subset[k].is_echo_state_property) continue;
						std::cerr << k << ",";
						for (i = 0; i < teacher_signals[TEST].size(); i++) {
							const double test_nmse = calc_nmse(teacher_signals[TEST][i], opt_w[k][i], output_node[k][TEST], unit_size, wash_out, step, false);
							if (i < task_size[0]) narma_task[k].push_back(test_nmse);
							else if (i < task_size[1]) approx_task[k].push_back(test_nmse);
							else if (i < task_size[2]) {
								const double tmp_L = 1.0 - test_nmse;
								if (tmp_L >= TRUNC_EPSILON) L[k] += tmp_L;
								sub_L[k].push_back(tmp_L);
							}
							else if (i < task_size[3]) {
								const double tmp_NL = (1.0 - test_nmse);
								if (tmp_NL >= TRUNC_EPSILON) NL[k] += tmp_NL;
								sub_NL[k].push_back(tmp_NL);
							}
							else if (i < task_size[4]) {
								if (d_vec[TEST].size() <= i - task_size[3]) {
									std::cerr << "error: " << d_vec[TEST].size() << "," << i - task_size[3] << std::endl;
									break;
								}
								int d_sum = 0;
								for (auto& e : d_vec[TEST][i - task_size[3]]) d_sum += e;
								const double tmp_NL = d_sum * (1.0 - test_nmse);
								if (tmp_NL >= TRUNC_EPSILON) {
									NL1_old[k] += 1.0 - test_nmse;
									NL_old[k] += tmp_NL;
								}
								sub_NL_old[k].push_back(tmp_NL);
							}
							else {
								std::cerr << "error" << std::endl;
							}
						}

					}
					std::cerr << "output..." << std::endl;
					// ファイル出力
					for (int k = 0; k < reservoir_subset.size(); k++) {
						if (!reservoir_subset[k].is_echo_state_property) continue;
						const double input_signal_factor = reservoir_subset[k].input_signal_factor;
						const double weight_factor = reservoir_subset[k].weight_factor;
						double bias_factor1 = reservoir_subset[k].bias_factor;
						if (bias_factor1 < 0) bias_factor1 = input_signal_factor * weight_factor;
						const double p = reservoir_subset[k].p;
						outputfile << toporogy_type << "," << function_name << "," << loop << "," << unit_size << "," << p << "," << input_signal_factor << "," << bias_factor1 << "," << weight_factor;
						outputfile << "," << L[k] << "," << NL[k] << "," << NL_old[k] << "," << NL1_old[k];

						for (int i = 0; i < std::min<int>(6, sub_NL_old[k].size()); i++) outputfile << "," << sub_NL_old[k][i];
						for (int i = 0; i < std::min<int>(51 - 2, sub_NL[k].size()); i++) outputfile << "," << sub_NL[k][i];
						for (int i = 0; i < std::min<int>(51 - 1, sub_L[k].size()); i++) outputfile << "," << sub_L[k][i];
						for (int i = 55 - 2; i < std::min<int>(101, sub_L[k].size()); i += 5) outputfile << "," << sub_L[k][i];
						for (int i = 0; i < narma_task[k].size(); i++) outputfile << "," << narma_task[k][i];
						for (int i = 0; i < approx_task[k].size(); i++)	outputfile << "," << approx_task[k][i];
						outputfile << std::endl;
					}
					reservoir_subset.clear();
				}
			}
		}
		outputfile.close();
	}
}