#include <random>
#define PI (3.14159265358979)
void generate_input_signal_random(std::vector<double>& input_signal, const int u_min, const int u_delta, const int step, const int seed);
void task_for_function_approximation(const std::vector<double>& input_signal, std::vector<double>& output_signal, const double nu, const int tau, const int step, const int seed);
void generate_narma_task(std::vector<double> input_signal, std::vector<double>& teacher_signal, const int tau, int step);
double t_tt_calc(std::vector<double> yt, const int wash_out, const int step);
double calc_mean_squared_average(const std::vector<double>& teacher_signal, const std::vector<double>& weight,
	const std::vector<std::vector<double>>& output_node, const int unit_size, const int wash_out, const int step, bool show);
double calc_nmse(const std::vector<double>& teacher_signal, const std::vector<double>& weight,
	const std::vector<std::vector<double>>& output_node, const int unit_size, const int wash_out, const int step, bool show);