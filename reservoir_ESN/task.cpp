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
	}
	return sum_squared_average / (step - wash_out);
}

double calc_nmse(const std::vector<double>& teacher_signal, const std::vector<double>& weight,
	const std::vector<std::vector<double>>& output_node, const int unit_size, const int wash_out, const int step, bool show) {
	return (calc_mean_squared_average(teacher_signal, weight, output_node, unit_size, wash_out, step) / t_tt_calc(teacher_signal, wash_out, step));
}