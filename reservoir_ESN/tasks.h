#pragma once
#include <random>
#include <algorithm>
#include <cassert>
#include <iostream>
#include <sstream>

#include "constant.h";
using uniform_real_dist_type = std::uniform_real_distribution<>;

class output_task {
public:
	std::vector<double> output_signal;
	double nmse;
	std::string task_name;
	std::string task_label;
};

class tasks {
public:
	int step;
	int seed;
	std::vector<double> input_signal;								// 入力信号
	std::vector<output_task> output_tasks;							// タスク
	std::mt19937 mt;
	std::vector<std::vector<int>> d_vec;
	tasks();
	tasks(int step, int seed);
	void generate_random_input(const int u_min, const int u_max);
	void generate_approx_task();
	void generate_narma_task();
	void generate_L_task(int max_L);
	void generate_NL_task();
private:
	void generate_d_sequence(std::vector<std::vector<int>>& d_vec, std::vector<int>& d, int d_sum_remain, int depth);
	void generate_d_vec();
};