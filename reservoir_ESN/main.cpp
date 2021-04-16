#include <iostream>
#include <vector>
#include <cmath>
#include <cblas.h>
#include <chrono>
#include <random>
#include "reservoir_layer.h"
#include "output_learning.h"
#include "task.h"
#define PHASE_NUM (3)
#define TRAIN (0)
#define VAL (1)
#define TEST (2)
double sinc(const double x) {
	if (x == 0) return 1.0;
	return sin(x) / x;
}
int main(void) {

	const int unit_size = 100;
	const int step = 5000;
	const int wash_out = 500;
	std::vector<std::vector<double>> input_signal(PHASE_NUM), teacher_signal(PHASE_NUM);
	for (int phase = 0; phase < PHASE_NUM; phase++) {
		generate_input_signal_random(input_signal[phase], -1.0, 2.0, step, phase);
		task_for_function_approximation(input_signal[phase], teacher_signal[phase], 1.0, 5, step, phase);
	}


	std::chrono::system_clock::time_point  start, end; // 型は auto で可
	for (int loop = 0; loop < 10; loop++) {
		for (int ite_p = 1; ite_p <= 10; ite_p++) {
			double opt_nmse = 1e+10;
			double opt_input_signal_factor = 0;
			double opt_weight_factor = 0;
			double opt_lm2 = 0;
			double test_nmse = 1e+10;
			std::vector<double> opt_w;
			for (int k = 0; k < 7 * 7; k++) {
				output_learning output_learning;
				const double p = ite_p * 0.1;
				const double input_signal_factor = ((k / 7) + 1) * 0.4;
				const double weight_factor = (k % 7 + 1) * 0.1;
				start = std::chrono::system_clock::now(); // 計測開始時間
				reservoir_layer reservoir_layer1(unit_size, unit_size / 10, input_signal_factor, weight_factor, p, sinc, loop, wash_out);
				reservoir_layer1.generate_reservoir();

				std::vector<std::vector<std::vector<double>>> output_node(PHASE_NUM, std::vector<std::vector<double>>(step + 10, std::vector<double>(unit_size + 1, 0)));
				bool ok = true;
				for (int phase = 0; phase < PHASE_NUM; phase++) {
					reservoir_layer1.reservoir_update(input_signal[phase], output_node[phase], step);
					if (!reservoir_layer1.is_echo_state_property(input_signal[phase])) ok = false;
				}
				if (!ok)
					continue;

				output_learning.generate_simultaneous_linear_equationsA(output_node[TRAIN], wash_out, step, unit_size);
				output_learning.generate_simultaneous_linear_equationsb(output_node[TRAIN], teacher_signal[TRAIN], wash_out, step, unit_size);
				double opt_lm = 0;
				double opt_lm_nmse = 1e+9;
				for (int i = 0; i < 20; i++) {
					for (int j = 0; j <= unit_size; j++) {
						output_learning.A[j][j] += pow(10, -16 + i * 0.5);
					}
					output_learning.IncompleteCholeskyDecomp2(unit_size + 1);
					double eps = 1e-12;
					int itr = 10;
					output_learning.ICCGSolver(unit_size + 1, itr, eps);
					const double nmse = calc_nmse(teacher_signal[VAL], output_learning.w, output_node[VAL], unit_size, wash_out, step, false);
					if (nmse < opt_lm_nmse) {
						opt_lm_nmse = nmse;
						opt_lm = i;
						if (opt_lm_nmse < opt_nmse) {
							opt_nmse = opt_lm_nmse;
							opt_input_signal_factor = input_signal_factor;
							opt_weight_factor = weight_factor;
							opt_lm2 = opt_lm;

							std::vector<std::vector<double>> output_node0(step + 10, std::vector<double>(unit_size + 1, 0));
							reservoir_layer1.reservoir_update(input_signal[TEST], output_node0, step);
							test_nmse = calc_nmse(teacher_signal[TEST], output_learning.w, output_node0, unit_size, wash_out, step, false);

						}
					}
				}

				end = std::chrono::system_clock::now();  // 計測終了時間
				double elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count(); //処理に要した時間をミリ秒に変換
				//std::cout << k << " " << p << " " << input_signal_factor << " " << weight_factor << " " << opt_lm << " " << opt_lm_nmse << " " << elapsed / 1000.0 << "[sec]" << std::endl;
				//std::cerr << calc_nmse(teacher_signal[TRAIN],opt_w, output_node[TRAIN], unit_size, wash_out, step) << std::endl;

			}

			std::cout << ite_p * 0.1 << "," << opt_input_signal_factor << "," << opt_weight_factor << "," << opt_lm2 << "," << opt_nmse << "," << test_nmse << std::endl;
		}
	}
}
