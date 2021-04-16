#include <iostream>
#include <vector>
#include <cmath>
#include <cblas.h>
#include "reservoir_layer.h"
#include "output_learning.h"

int main(void) {
	output_learning output_learning;
	for (int k = 1; k <= 20; k++) {
		reservoir_layer reservoir_layer(100, 10, 1.0, k * 0.1, 0.8, sin, 0, 500);
		reservoir_layer.generate_reservoir();
		std::vector<double> input_signal, teacher_signal;
		for (int i = 0; i < 1000; i++) {
			input_signal.push_back(cos(i/10.0));
			teacher_signal.push_back(sin(i / 10.0));
		}
		std::vector<std::vector<double>> output_node(2000, std::vector<double>(200, 0));
		reservoir_layer.reservoir_update(input_signal, output_node, 1000);
		if (!reservoir_layer.is_echo_state_property(input_signal)) continue;
		std::vector<std::vector<double>> L(210, std::vector<double>(210, 0.0));
		std::vector<double> d(210, 0.0);
		output_learning.generate_simultaneous_linear_equationsA(output_node, 500, 1000, 100);
		output_learning.generate_simultaneous_linear_equationsb(output_node, teacher_signal, 500, 1000, 100);
		for (int i = 0; i < 1; i++) {
			for (int j = 0; j < 100; j++) {
				output_learning.A[j][j] += 1e-12;
			}
			output_learning.IncompleteCholeskyDecomp2(L, d, 100);
			double eps = 1e-12;
			int itr = 100;
			output_learning.ICCGSolver(L, d, 100, itr, eps);
			std::cout << eps << " " << itr << std::endl;
			double error_sum = 0.0;
			for (int t = 500; t < 1000; t++) {
				const double reservoir_predict_signal = cblas_ddot(100 + 1, output_learning.w.data(), 1, output_node[t + 1].data(), 1);
				if (t == 510) {
					std::cout << reservoir_predict_signal << " " << teacher_signal[t] << std::endl;
				}
				const double tmp = reservoir_predict_signal - teacher_signal[t];
				error_sum += tmp * tmp;
			}
			std::cout << error_sum << std::endl;
		}
		std::cout << k << std::endl;
	}
}