#include <random>
#include <algorithm>
#include <cassert>
#include <iostream>
#include <sstream>
#include "tasks.h"
#include "constant.h"

template <typename T>
std::string to_string_with_precision(const T a_value, const int n = 6)
{
	std::ostringstream out;
	out.precision(n);
	out << std::fixed << a_value;
	return out.str();
}
tasks::tasks(int step, int seed) {
	this->step = step;
	this->seed = seed;
	input_signal.resize(step);
}

// [u_min, u_max]‚Ì”ÍˆÍ‚Åˆê—l—”‚ğ¶¬‚·‚é
void tasks::generate_random_input(const int u_min, const int u_max) {
	uniform_real_dist_type::param_type param(u_min, u_max);
	uniform_real_dist_type dist(param);
	for (int t = 0; t < step; t++) {
		input_signal[t] = dist(mt);
	}
}

void tasks::generate_approx_task(const std::vector<double>& tau_set, const std::vector<double>& nu_set) {
	for (auto tau : tau_set) {
		for (auto nu : nu_set) {
			output_task task;
			int tmp_tau = (int)(pow(2, tau) + 0.5);
			task.task_name = "approx_" + std::to_string(tmp_tau) + "_" + to_string_with_precision(nu, 1);
			task.task_label = "approx";
			double tmp_nu = nu = pow(2, nu);
			
			task.output_signal.resize(step);
			for (int t = 0; t < step; t++) {
				double sum = 0.0;
				for (int r = std::max<int>(0, t - tmp_tau); r <= t; r++) sum += input_signal[r];
				double average = sum / sqrt((tmp_tau + 1.0));
				task.output_signal[t] = sin(tmp_nu * PI * average + 0.25 * PI);
			}
			output_tasks.push_back(task);
		}
	}
}

void tasks::generate_narma_task(std::vector<int>& tau_set) {
	for (auto tau : tau_set) {
		std::vector<double> input_signal = this->input_signal;
		output_task task;
		task.output_signal.resize(step);
		task.task_name = "narma_" + (std::to_string(tau));
		task.task_label = "narma";
		const double alpha = 0.3;
		const double beta = 0.05;
		const double gamma = 1.5;
		double delta = 0.1;
		tau--;
		if (tau >= 10) delta = 0.01;
		for (int t = 0; t < step; t++) {
			input_signal[t] = (input_signal[t] + 1) / 4;
		}
		for (int t = 0; t < step; t++) {
			double sum = 0.0;
			if (t - tau - 1 >= 0) {
				for (int i = tau + 1; i >= 1; i--) {  // for(i=tau; i>=1; i--){
					sum = sum + task.output_signal[t - i];
				}

				task.output_signal[t] =
					alpha * task.output_signal[t - 1] + beta * task.output_signal[t - 1] * sum + gamma * input_signal[t - tau - 1] * input_signal[t] + delta;
				if (tau > 9) task.output_signal[t] = tanh(task.output_signal[t]);  // NARMA(tau>=10)
			}
			else
				task.output_signal[t] = 0;

			//... cutt-off bound ...
			if (task.output_signal[t] > 1.0) {
				task.output_signal[t] = 1.0;
			}
			else if (task.output_signal[t] < -1.0) {
				task.output_signal[t] = -1.0;
			}
		}
		output_tasks.push_back(task);
	}
}
void tasks::generate_L_task(int max_L) {
	for (int tau = 1; tau <= max_L; tau++) {
		output_task task;
		task.task_label = "L";
		task.task_name = "L_" + std::to_string(tau);
		task.output_signal.resize(step);
		for (int t = 0; t < step; t++) {
			if (t - tau >= 0)
				task.output_signal[t] = input_signal[t - tau];
			else
				task.output_signal[t] = 0.0;
		}
		output_tasks.push_back(task);
	}
}

void tasks::generate_NL_task() {
	std::vector<std::vector<int>> d_vec = tasks::generate_d_vec();
	for (int i = 0; i < d_vec.size(); i++) {
		output_task task;
		task.task_label = "NL";
		task.task_name = "NL_" + std::to_string(i);
		task.output_signal.resize(step);
		std::vector<int> d = d_vec[i];
		for (int t = 0; t < step; t++) {
			double x = 1.0;
			for (int i = 0; i < d.size(); i++) {
				if (t - (i + 1) >= 0) {
					assert(-1.0 < input_signal[t - i] && input_signal[t - i] < 1.0);
					x *= std::legendre(d[i], input_signal[t - i]);
				}
			}
			task.output_signal[t] = x;
		}
		output_tasks.push_back(task);
	}
}
void tasks::generate_d_sequence(std::vector<std::vector<int>>& d_vec, std::vector<int>& d, int d_sum_remain, int depth = 0) {
	if (d_sum_remain <= 0) {
		d_vec.push_back(d);
		return;
	}
	d[depth]++;
	generate_d_sequence(d_vec, d, d_sum_remain - 1, depth);
	d[depth]--;
	if (depth + 1 < d.size()) generate_d_sequence(d_vec, d, d_sum_remain, depth + 1);
}

std::vector<std::vector<int>> tasks::generate_d_vec() {
	std::vector<std::vector<int>> d_vec;
	std::vector<int> d;
	d.resize(12); generate_d_sequence(d_vec, d, 2);
	d.resize(8); generate_d_sequence(d_vec, d, 3);
	d.resize(6); generate_d_sequence(d_vec, d, 4);
	d.resize(4); generate_d_sequence(d_vec, d, 5);
	d.resize(3); generate_d_sequence(d_vec, d, 6);
	d.resize(3); generate_d_sequence(d_vec, d, 7);
	return d_vec;
}