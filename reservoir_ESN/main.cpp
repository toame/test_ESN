#include <iostream>
#include <cmath>
#include "reservoir_layer.h"
#include "output_learning.h"

int main(void) {
	reservoir_layer reservoir_layer(100, 10, 1.0, 1.0, 0.8, sin, 0, 500);

}