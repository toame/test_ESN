#pragma once
#include "task.h"
#include "constant.h"

void generate_d_sequence(std::vector<std::vector<int>>& d_vec, std::vector<int>& d, int d_sum_remain, int depth = 0) {
	if (d_sum_remain <= 0) {
		d_vec.push_back(d);
		// for(auto& e: d) std::cerr << e << " ";
		// std::cerr << std::endl;
		return;
	}
	// std::cerr << d_sum_remain << " " << depth << std::endl;
	d[depth]++;
	generate_d_sequence(d_vec, d, d_sum_remain - 1, depth);
	d[depth]--;
	if (depth + 1 < d.size()) generate_d_sequence(d_vec, d, d_sum_remain, depth + 1);
}

void generate_d_sequence_set(std::vector<std::vector<std::vector<int>>>& d_vec) {
	for (int mode = 0; mode < PHASE_NUM; mode++) {
		std::vector<int> d;
		d.resize(12); generate_d_sequence(d_vec[mode], d, 2);
		d.resize(8); generate_d_sequence(d_vec[mode], d, 3);
		d.resize(6); generate_d_sequence(d_vec[mode], d, 4);
		d.resize(4); generate_d_sequence(d_vec[mode], d, 5);
		d.resize(3); generate_d_sequence(d_vec[mode], d, 6);
		d.resize(3); generate_d_sequence(d_vec[mode], d, 7);
		d.resize(2); generate_d_sequence(d_vec[mode], d, 8);
		d.resize(2); generate_d_sequence(d_vec[mode], d, 9);
		d.resize(2); generate_d_sequence(d_vec[mode], d, 10);
	}
}

inline double squared(const double x) {
	return x * x;
}

double t_tt_calc(std::vector<double> yt, const int wash_out, const int step) {
	double t_ave0 = 0.0, tt_ave0 = 0.0;
	for (int t = wash_out + 1; t < step; t++) {
		t_ave0 += yt[t];
		tt_ave0 += yt[t] * yt[t];
	}
	t_ave0 /= (step - wash_out);
	tt_ave0 /= (step - wash_out);

	return tt_ave0 - t_ave0 * t_ave0;
}

double calc_mean_squared_average(const std::vector<double>& teacher_signal, const std::vector<double>& weight,
	const std::vector<std::vector<double>>& output_node, const int unit_size, const int wash_out, const int step, bool show, std::string name) {
	double sum_squared_average = 0.0;
	//std::ofstream outputfile("output_predict/" + name + ".txt", std::ios::app);
	//if(show)
	//	outputfile << "t,predict_test,teacher" << std::endl;
	assert(output_node.size() == step + 1);
	assert(weight.size() == unit_size + 1);
	for (int t = wash_out + 1; t < step; t++) {
		assert(output_node[t + 1].size() == unit_size + 1);
	}
	
	for (int t = wash_out + 1; t < step; t++) {
		const double reservoir_predict_signal = cblas_ddot(unit_size + 1, weight.data(), 1, output_node[t + 1].data(), 1);
		//double reservoir_predict_signal = 0.0;
		//for (int n = 0; n <= unit_size; n++) {
		//	reservoir_predict_signal += weight[n] * output_node[t + 1][n];
		//}
		sum_squared_average += squared(teacher_signal[t] - reservoir_predict_signal);
		//if (show) {
		//	outputfile << t << "," << reservoir_predict_signal << "," << teacher_signal[t] << "," << sum_squared_average << std::endl;
		//}
	}
	return sum_squared_average / (step - wash_out - 1);
}

double calc_nmse(const std::vector<double>& teacher_signal, const std::vector<double>& weight,
	const std::vector<std::vector<double>>& output_node, const int unit_size, const int wash_out, const int step, bool show, std::string name) {
	const double tmp1 = calc_mean_squared_average(teacher_signal, weight, output_node, unit_size, wash_out, step, show, name);
	const double tmp2 = t_tt_calc(teacher_signal, wash_out, step);
	return tmp1 / tmp2;
}
double calc_mean_squared_average_fast(const std::vector<double>& teacher_signal, const std::vector<double>& weight,
	const std::vector<double>& output_node, const int unit_size, const int wash_out, const int step, bool show, std::string name) {
	double sum_squared_average = 0.0;
	std::vector<double> reservoir_predict_signal(step);
	const double alpha = 1.0, beta = 0.0;
	cblas_dgemv(CblasRowMajor, CblasNoTrans, step, unit_size + 1, alpha, output_node.data(), unit_size + 1, weight.data(), 1, beta, reservoir_predict_signal.data(), 1);
	for (int t = wash_out + 1; t < step; t++) {
		sum_squared_average += squared(teacher_signal[t] - reservoir_predict_signal[t]);
	}
	return sum_squared_average / (step - wash_out - 1);
}
double calc_nmse_fast(const std::vector<double>& teacher_signal, const std::vector<double>& weight,
	const std::vector<double>& output_node, const int unit_size, const int wash_out, const int step, bool show, std::string name) {
	return (calc_mean_squared_average_fast(teacher_signal, weight, output_node, unit_size, wash_out, step, show, name) / t_tt_calc(teacher_signal, wash_out, step));
}