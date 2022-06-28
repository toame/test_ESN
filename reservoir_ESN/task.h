#pragma once
#include <vector>
#include <cblas.h>
#include <iostream>
#include <cassert>
double t_tt_calc(std::vector<double> yt, const int wash_out, const int step);
double calc_mean_squared_average(const std::vector<double>& teacher_signal, const std::vector<double>& weight,
	const std::vector<std::vector<double>>& output_node, const int unit_size, const int wash_out, const int step, bool show = false, std::string name = "");
double calc_nmse(const std::vector<double>& teacher_signal, const std::vector<double>& weight,
	const std::vector<std::vector<double>>& output_node, const int unit_size, const int wash_out, const int step, bool show = false, std::string name = "");
double calc_nmse_fast(const std::vector<double>& teacher_signal, const std::vector<double>& weight,
	const std::vector<double>& output_node, const int unit_size, const int wash_out, const int step, bool show = false, std::string name = "");