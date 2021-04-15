#include <iostream>
#include <vector>
#include <cmath>
#include "reservoir_layer.h"
#include "output_learning.h"

int main(void) {
	for (int k = 1; k <= 20; k++) {
		reservoir_layer reservoir_layer(100, 10, 1.0, k * 0.1, 0.8, sin, 0, 500);
		reservoir_layer.generate_reservoir();
		std::vector<double> input_signal;
		for (int i = 0; i < 1000; i++) {
			input_signal.push_back(sin(i / 10.0));
		}
		std::vector<std::vector<double>> output_signal(2000, std::vector<double>(200, 0));
		reservoir_layer.reservoir_update(input_signal, output_signal, 1000);
		std::cout << k * 0.1 << " " << reservoir_layer.is_echo_state_property(input_signal) << std::endl;
	}
}