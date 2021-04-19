#include "task.h"
#include <cblas.h>
#include <iostream>
void generate_input_signal_random(std::vector<double>& input_signal, const int u_min, const int u_delta, const int step, const int seed) {
	std::mt19937 mt(seed);
	std::uniform_real_distribution<> rand_0to1(0, 1);
	std::uniform_int_distribution<> rand_0or1(0, 1);
	for (int t = 0; t < step; t++) input_signal.push_back(u_min + u_delta * rand_0to1(mt));
}

void task_for_function_approximation(const std::vector<double>& input_signal, std::vector<double>& output_signal, const double nu, const int tau,
	const int step, const int seed) {
	std::mt19937 mt(seed);
	std::uniform_real_distribution<> rand_0to1(0, 1);
	std::uniform_int_distribution<> rand_0or1(0, 1);

	for (int t = 0; t < step; t++) {
		if (t - tau >= 0)
			output_signal.push_back(sin(nu * PI * input_signal[t - tau]));
		else
			output_signal.push_back(0);
	}
}
//  0.3, 0.05, 1.5, 0.1
void generate_narma_task(std::vector<double> input_signal, std::vector<double>& teacher_signal, const int tau, int step) {
	const double alpha = 0.3;
	const double beta = 0.05;
	const double gamma = 1.5;
	const double delta = 0.1;
	for (int t = 0; t < step; t++) input_signal[t] = (input_signal[t] + 1) / 4;
	teacher_signal.resize(step);
	for (int t = 0; t < step; t++) {
		double sum = 0.0;
		if (t - tau >= 0) {
			for (int i = tau + 1; i >= 1; i--) {  // for(i=tau; i>=1; i--){
				sum = sum + teacher_signal[t - i];
			}

			teacher_signal[t] =
				alpha * teacher_signal[t - 1] + beta * teacher_signal[t - 1] * sum + gamma * input_signal[t - tau] * input_signal[t] + delta;
			if (tau > 9) teacher_signal[t] = tanh(teacher_signal[t]);  // NARMA(tau>=10)
		}
		else
			teacher_signal[t] = 0;

		//... cutt-off bound ...
		if (teacher_signal[t] > 1.0) {
			teacher_signal[t] = 1.0;
		}
		else if (teacher_signal[t] < -1.0) {
			teacher_signal[t] = -1.0;
		}
	}
}

inline double squared(const double x) {
	return x * x;
}

double t_tt_calc(std::vector<double> yt, const int wash_out, const int step) {
	double t_ave0 = 0.0, tt_ave0 = 0.0;
	for (int t = wash_out + 1; t <= step; t++) {
		t_ave0 += yt[t];
		tt_ave0 += yt[t] * yt[t];
	}
	t_ave0 /= (step - wash_out);
	tt_ave0 /= (step - wash_out);
	return tt_ave0 - t_ave0 * t_ave0;
}

double calc_mean_squared_average(const std::vector<double>& teacher_signal, const std::vector<double>& weight,
	const std::vector<std::vector<double>>& output_node, const int unit_size, const int wash_out, const int step, bool show = false) {
	double sum_squared_average = 0.0;
	
	for (int t = wash_out + 1; t <= step; t++) {
		//const double reservoir_predict_signal = cblas_ddot(unit_size + 1, weight.data(), 1, output_node[t + 1].data(), 1);
		double reservoir_predict_signal = 0.0;
		for (int n = 0; n <= unit_size; n++) {
			reservoir_predict_signal += weight[n] * output_node[t + 1][n];
		}
		sum_squared_average += squared(teacher_signal[t] - reservoir_predict_signal);
		if (show) {
			std::cout << t << " " << reservoir_predict_signal << " " << teacher_signal[t] << std::endl;
		}
	}
	return sum_squared_average / (step - wash_out);
}

double calc_nmse(const std::vector<double>& teacher_signal, const std::vector<double>& weight,
	const std::vector<std::vector<double>>& output_node, const int unit_size, const int wash_out, const int step, bool show) {
	return (calc_mean_squared_average(teacher_signal, weight, output_node, unit_size, wash_out, step, show) / t_tt_calc(teacher_signal, wash_out, step));
}