#include <iostream>
#include <vector>
#include <cmath>
#include <cblas.h>
#include <chrono>
#include "reservoir_layer.h"
#include "output_learning.h"
void test();
int main(void) {
	//test();
	output_learning output_learning;
	const int unit_size = 500;
	const int step = 5000;
	const int wash_out = 500;
	std::vector<double> input_signal, teacher_signal;
	for (int i = 0; i < step; i++) {
		input_signal.push_back(cos(i / 10.0));
		teacher_signal.push_back(sin(i / 10.0));
	}
	for (int p = 5; p <= 5; p++) {
		std::chrono::system_clock::time_point  start, end; // 型は auto で可
		for (int k = 0; k < 100; k++) {
			start = std::chrono::system_clock::now(); // 計測開始時間
			reservoir_layer reservoir_layer(unit_size, unit_size / 10, (k / 10 + 1) * 0.1, (k % 10 + 1) * 0.1, p * 0.1, sin, 0, wash_out);
			reservoir_layer.generate_reservoir();

			std::vector<std::vector<double>> output_node(step + 10, std::vector<double>(unit_size + 1, 0));

			reservoir_layer.reservoir_update(input_signal, output_node, step);
			if (!reservoir_layer.is_echo_state_property(input_signal)) continue;

			output_learning.generate_simultaneous_linear_equationsA(output_node, wash_out, step, unit_size);
			output_learning.generate_simultaneous_linear_equationsb(output_node, teacher_signal, wash_out, step, unit_size);

			for (int i = 0; i < 20; i++) {
				for (int j = 0; j <= unit_size; j++) {
					output_learning.A[j][j] += pow(10, -16 + i * 0.5);
				}
				output_learning.IncompleteCholeskyDecomp2(unit_size + 1);
				double eps = 1e-18;
				int itr = 100;
				output_learning.ICCGSolver(unit_size + 1, itr, eps);
				double error_sum = 0.0;
				for (int t = wash_out; t < step; t++) {
					const double reservoir_predict_signal = cblas_ddot(unit_size + 1, output_learning.w.data(), 1, output_node[t + 1].data(), 1);
					const double tmp = reservoir_predict_signal - teacher_signal[t];
					error_sum += tmp * tmp;
				}
				//std::cout << k << " " << pow(10, -17 + i) << " " << error_sum << std::endl;
			}
			
			end = std::chrono::system_clock::now();  // 計測終了時間
			double elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count(); //処理に要した時間をミリ秒に変換
			std::cout << k << " " << elapsed /1000.0 << "[sec]" << std::endl;
		}
	}
}
