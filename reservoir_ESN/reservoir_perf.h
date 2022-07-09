#include <algorithm>
#include "tasks.h"
#include "constant.h"
#include "reservoir_layer.h"
class reservoir_perf {
public:
	reservoir_layer reservoir;
	tasks _tasks;
	double L;
	double L_cut;
	double NL;
	double NL_old;
	double NL_old_cut1;
	double NL_old_cut2;
	double NL1_old;
	std::vector<double> sub_L;
	std::map<int, double> sub_NL_old2;
	std::vector<double> sub_NL_old;
	std::vector<double> narma_task;
	std::vector<double> approx_task;
	std::vector<double> henon_task;
	std::vector<double> laser_task;
	std::vector<double> count_task;
	int maxL;
	reservoir_perf() {
		L = 0.0;
		L_cut = 0.0;
		NL = 0.0;
		NL_old = 0.0;
		NL1_old = 0.0;
		NL_old_cut1 = 0.0;
		NL_old_cut2 = 0.0;
		maxL = 0;
		sub_NL_old.resize(10);
	}
	reservoir_perf(tasks _tasks) {
		this->_tasks = _tasks;
		L = 0.0;
		L_cut = 0.0;
		NL = 0.0;
		NL_old = 0.0;
		NL1_old = 0.0;
		NL_old_cut1 = 0.0;
		NL_old_cut2 = 0.0;
		maxL = 0;
		sub_NL_old.resize(10);
	}
	void add_task(tasks _tasks) {
		for (int i = 0; i < _tasks.output_tasks.size(); i++) {
			const output_task& task = _tasks.output_tasks[i];
			if (task.task_label == "narma") narma_task.push_back(task.nmse);
			else if (task.task_label == "approx") approx_task.push_back(task.nmse);
			else if (task.task_label == "L") {
				calc_L(task.nmse, task.task_name);
			}
			else if (task.task_label == "NL2") {
				const double tmp_NL = (1.0 - task.nmse);
			}
			else if (task.task_label == "NL") {
				int idx = stoi(task.task_name.substr(3));
				std::vector<int> d = _tasks.d_vec[idx];
				calc_NL(d, task.nmse, task.task_name);
			}
			else if (task.task_label == "henon") {
				henon_task.push_back(task.nmse);
			}
			else if (task.task_label == "laser") {
				laser_task.push_back(task.nmse);
			}
			else if (task.task_label == "count") {
				count_task.push_back(task.nmse);
			}
			else {
				std::cerr << "error" << std::endl;
			}
		}
	}

	void calc_L(double nmse, std::string task_name) {
		int tau = stoi(task_name.substr(2));
		double tmp_L = 1.0 - nmse;
		if (tmp_L > TRUNC_EPSILON) {
			L += tmp_L;
			L_cut += tmp_L * tmp_L;
			if (tmp_L >= 0.9) {
				maxL = std::max(tau, maxL);
			}
		}
		sub_L.push_back(tmp_L);
	}
	void calc_NL(std::vector<int>& d, double nmse, std::string task_name) {
		int idx = stoi(task_name.substr(3));
		int d_sum = 0;
		int last = 0;
		for (int r = 0; r < d.size(); r++) {
			d_sum += d[r];
			if (d[r] > 0) last = r;
		}
		const double tmp_NL = d_sum * (1.0 - nmse);
		const double tmp_NL1 = (1.0 - nmse);
		sub_NL_old2[idx] = 0;
		if (tmp_NL >= TRUNC_EPSILON) {
			NL1_old += tmp_NL1;
			NL_old += tmp_NL;
			sub_NL_old[d_sum] += tmp_NL1 * tmp_NL1 * d_sum;
			NL_old_cut1 += tmp_NL1 * tmp_NL1 * d_sum;
			sub_NL_old2[idx] = tmp_NL1;
			if (last <= maxL) {
				NL_old_cut2 += tmp_NL;
			}
		}
	}

	void reservoir_perf_output(std::ofstream& outputfile) {
		const double input_signal_factor = reservoir.input_signal_factor;
		const double input_gain = reservoir.input_gain;
		const double feed_gain = reservoir.feed_gain;
		const int seed = reservoir.seed;
		const double p = reservoir.p;
		std::string function_name = reservoir.nonlinear_name;
		outputfile << reservoir.toporogy_type << ","
			<< function_name << ","
			<< seed << ","
			<< reservoir.unit_size << ","
			<< p << ","
			<< input_signal_factor << ","
			<< input_gain << ","
			<< feed_gain << ","
			<< reservoir.d;
		outputfile << "," << L << "," << L_cut << "," << NL << "," << NL_old << "," << NL1_old << "," << NL_old_cut1 << "," << NL_old_cut2;

		for (int i = 2; i <= 10; i++) outputfile << "," << sub_NL_old[i];
		//for (int i = 0; i < sub_NL.size(); i++) outputfile << "," << sub_NL[k][i];
		for (int i = 0; i < sub_L.size(); i++) outputfile << "," << sub_L[i];
		for (int i = 0; i < sub_NL_old2.size(); i++) outputfile << "," << sub_NL_old2[i];
		for (int i = 0; i < approx_task.size(); i++)	outputfile << "," << approx_task[i];
		for (int i = 0; i < narma_task.size(); i++) outputfile << "," << narma_task[i];
		for (int i = 0; i < henon_task.size(); i++) outputfile << "," << henon_task[i];
		for (int i = 0; i < laser_task.size(); i++) outputfile << "," << laser_task[i];
		for (int i = 0; i < count_task.size(); i++) outputfile << "," << count_task[i];
		outputfile << std::endl;
	}
};
