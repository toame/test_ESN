#include <string>
#include <sstream>
#include <vector>
#include <fstream>
#pragma once
#include "task.h"
#include "tasks.h"


template <typename T>
std::string to_string_with_precision(const T a_value, const int n = 6)
{
	std::ostringstream out;
	out.precision(n);
	out << std::fixed << a_value;
	return out.str();
}

void output_NL_format(tasks task) {
	std::ofstream output_NL_d("output_data/d_vec.csv");
	auto d_vec = task.d_vec;
	for (auto& d : d_vec) {
		for (int i = 0; i < d.size(); i++) {
			if (i != 0) output_NL_d << ",";
			output_NL_d << d[i];
		}
		output_NL_d << std::endl;
	}
	output_NL_d.close();
}