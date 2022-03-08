#include <algorithm>
#include "tasks.h"
#include "constant.h"
class reservoir_perf {
public:
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
	void calc_NL(std::vector<int>&d, double nmse, std::string task_name) {
		int idx = stoi(task_name.substr(3));
		int d_sum = 0;
		int last = 0;
		for (int r = 0; r < d.size(); r++) {
			d_sum += d[r];
			if (d[r] > 0) last = r;
		}
		const double tmp_NL = d_sum * (1.0 - nmse);
		const double tmp_NL1 = (1.0 - nmse);
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

};
