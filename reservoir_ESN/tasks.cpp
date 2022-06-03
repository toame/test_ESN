#include "tasks.h"
#include <fstream>
template <typename T>
std::string to_string_with_precision(const T a_value, const int n = 6)
{
	std::ostringstream out;
	out.precision(n);
	out << std::fixed << a_value;
	return out.str();
}
tasks::tasks() {
	seed = 0;
	step = 4000;
	input_signal.resize(step);
	d_vec.resize(PHASE_NUM);
}
tasks::tasks(int step, int seed) {
	this->step = step;
	this->seed = seed;
	input_signal.resize(step);
	d_vec.resize(PHASE_NUM);
}

// [u_min, u_max]‚Ì”ÍˆÍ‚Åˆê—l—”‚ğ¶¬‚·‚é
void tasks::generate_random_input(const int u_min, const int u_max) {
	uniform_real_dist_type::param_type param(u_min, u_max);
	uniform_real_dist_type dist(param);
	for (int t = 0; t < step; t++) {
		input_signal[t] = dist(mt);
	}
}

void tasks::generate_henon_input_output() {
	double a = 0.1, b = 0.2, c = 0;
	const double alpha = 1.4;
	const double beta = 0.3;
	std::vector < double> input_signal_tmp;
	input_signal_tmp.resize(step + 100);
	for (int t = 0; t < seed * 5000; t++) {
		c = 1 - alpha * b * b + beta * a;
		std::swap(a, b);
		std::swap(b, c);
	}
	input_signal_tmp[0] = a;
	input_signal_tmp[1] = b;
	for (int t = 2; t < step + 100; t++) {
		input_signal_tmp[t] = 1 - alpha * input_signal_tmp[t - 1] * input_signal_tmp[t - 1] + beta * input_signal_tmp[t - 2];
	}
	for (int t = 0; t < step; t++) {
		input_signal[t] = input_signal_tmp[t];
	}
	
	std::vector<int> fsteps({ 1, 2, 3, 4, 5, 6, 7, 8 });
	for (auto fstep : fsteps) {
		output_task task;
		task.task_name = "henon_" + std::to_string(fstep);
		task.task_label = "henon";
		task.output_signal.resize(step);
		for (int t = 0; t < step; t++) {
			task.output_signal[t] = input_signal_tmp[t + fstep];
		}
		output_tasks.push_back(task);
	}
}
void tasks::generate_count_input_output() {
	for (int t = 0; t < step; t++) {
		input_signal[t] = (mt() % 2) * 2 - 1.0;
	}
	std::vector<int> fsteps({ 2, 3, 4, 5, 6, 7, 8, 9, 10 });
	for (auto fstep : fsteps) {
		output_task task;
		task.task_name = "count_" + std::to_string(fstep);
		task.task_label = "count";
		task.output_signal.resize(step);
		for (int t = 0; t < step; t++) {
			task.output_signal[t] = 1.0;
			for (int i = std::max<int>(0, t - fstep + 1); i <= t; i++) {
				task.output_signal[t] *= input_signal[i];
			}
		}
		output_tasks.push_back(task);
	}
}
void tasks::generate_laser_input_output(int k) {
	std::ifstream ifs("santafe.dat");

	std::string line;
	int cnt = 0;
	std::vector <double> input_signal_tmp;
	while (std::getline(ifs, line)) {
		cnt--;
		if (cnt <= 0) {
			double num = std::stoi(line) / 255.0;
			//std::cerr << num << std::endl;
			input_signal_tmp.push_back(num);
		}
	}
	for (int t = 0; t < step; t++) {
		input_signal[t] = input_signal_tmp[t + step * ((seed + k) % 3)];
	}
	std::vector<int> fsteps({ 1, 2, 3, 4, 5, 6, 7, 8, 9, 10 });
	for (auto fstep : fsteps) {
		output_task task;
		task.task_name = "laser" + std::to_string(k) + "_" + std::to_string(fstep);
		task.task_label = "laser";
		task.output_signal.resize(step);
		for (int t = 0; t < step; t++) {
			task.output_signal[t] = input_signal[t + fstep];
		}
		output_tasks.push_back(task);
	}
}

void tasks::generate_approx_task() {
	std::vector<double> approx_nu_set({ -3.0, -2.5, -2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0 });
	std::vector<double> approx_tau_set({ -2, 0, 1, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0 });
	for (auto tau : approx_tau_set) {
		for (auto nu : approx_nu_set) {
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

void tasks::generate_narma_task() {
	std::vector<int> narma_tau_set({ 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20 });
	for (auto tau : narma_tau_set) {
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

double chebyshev(int i, double x) {
	if (i == 0) return 1.0;
	else if (i == 1) return x;
	else if (i == 2) return 2.0 * x * x - 1.0;
	else if (i == 3) return 4.0 * x * x * x - 3.0 * x;
	else if (i == 4) return 8.0 * pow(x, 4.0) - 8.0 * pow(x, 2.0) + 1.0;
	else if (i == 5) return 16.0 * pow(x, 5.0) - 20.0 * pow(x, 3.0) + 5.0 * x;
	else if (i == 6) return 32.0 * pow(x, 6.0) - 48.0 * pow(x, 4.0) + 18.0 * pow(x, 2.0) - 1.0;
	else if (i == 7) return 64.0 * pow(x, 7.0) - 112.0 * pow(x, 5.0) + 56.0 * pow(x, 3.0) - 7.0 * x;
	else if (i == 8) return 128.0 * pow(x, 8.0) - 256.0 * pow(x, 6.0) + 160.0 * pow(x, 4.0) - 32.0 * pow(x, 2.0) + 1.0;
	else if (i == 9) return 256.0 * pow(x, 9.0) - 576.0 * pow(x, 7.0) + 432.0 * pow(x, 5.0) - 120.0 * pow(x, 3.0) + 9.0 * x;
	else if (i == 10) return 512.0 * pow(x, 10.0) - 1280.0 * pow(x, 8.0) + 1120.0 * pow(x, 6.0) - 400.0 * pow(x, 4.0) + 50.0 * pow(x, 2.0) - 1.0;
	else if (i == 11) return 1024.0 * pow(x, 11.0) - 2816.0 * pow(x, 9.0) + 2816.0 * pow(x, 7.0) - 1232.0 * pow(x, 5.0) + 220.0 * pow(x, 3.0) - 11.0 * x;
	return 1.0;
}

void tasks::generate_NL_task() {
	tasks::generate_d_vec();
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
					//x *= std::legendre(d[i], input_signal[t - i]);
					x *= chebyshev(d[i], input_signal[t - i]);
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

void tasks::generate_d_vec() {
	std::vector<int> d;
	d.resize(12); generate_d_sequence(d_vec, d, 2);
	d.resize(8); generate_d_sequence(d_vec, d, 3);
	d.resize(6); generate_d_sequence(d_vec, d, 4);
	d.resize(4); generate_d_sequence(d_vec, d, 5);
	d.resize(3); generate_d_sequence(d_vec, d, 6);
	d.resize(3); generate_d_sequence(d_vec, d, 7);
	d.resize(2); generate_d_sequence(d_vec, d, 8);
	d.resize(2); generate_d_sequence(d_vec, d, 9);
	d.resize(2); generate_d_sequence(d_vec, d, 10);
}
