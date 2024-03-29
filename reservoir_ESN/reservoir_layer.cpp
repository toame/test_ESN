﻿#include "reservoir_layer.h"
#include "constant.h"
double sinc(const double x) {
	if (x == 0) return 1.0;
	return sin(PI * x) / (PI * x);
}
double gauss(double y) { return exp(-y * y / (2.0 * 0.4 * 0.4)) / (sqrt(PI * 2) * 0.4); }
double oddsinc(double y) {
	if (y <= 0) return sin(PI * y) / (PI * (y + 1));
	else return sin(PI * -y) / (PI * (y - 1));
}
reservoir_layer::reservoir_layer() {}
reservoir_layer::reservoir_layer(const int unit_size, const int connection_degree, const double iss_factor, const double weight_factor, const double bias_factor, const double p,
		std::string nonlinear_name, const unsigned int seed, const int wash_out, const std::string toporogy_type) {
	this->unit_size = unit_size;
	this->connection_degree = connection_degree;
	this->input_signal_factor = iss_factor;
	this->weight_factor = weight_factor;
	this->bias_factor = bias_factor;
	this->p = p;
	this->seed = seed;
	this->nonlinear_name = nonlinear_name;
	this->wash_out = wash_out;
	this->toporogy_type = toporogy_type;
	if (nonlinear_name == "sinc") nonlinear = sinc;
	else if (nonlinear_name == "tanh") nonlinear = tanh;
	else assert(false);
	if (bias_factor < -0.9) this->bias_factor = input_signal_factor * weight_factor;
	node_type.resize(unit_size + 1);
	adjacency_list.resize(unit_size + 1, std::vector<int>(connection_degree + 1));
	weight_reservoir.resize(unit_size + 1, std::vector<double>(connection_degree + 1));
	input_signal_strength.resize(unit_size + 1);
	mt.seed(seed);
}

std::vector<reservoir_layer> reservoir_layer::generate_reservoir(const int unit_size, const unsigned int loop, const int wash_out) {
	std::vector<reservoir_layer> ret;


	std::vector<std::string> nonlinear_vec{ "sinc", "tanh" };
	std::vector<std::string> toporogy_types{ "random", "ring", "doubly_ring", "sparse_random" };
	std::vector<double> p_set{ 0.05, 0.1, 0.2, 0.35, 0.5, 0.65, 0.8, 0.9, 0.95, 1.0, 0.0 };
	std::vector<double> bias_set{ 0, 1, 2, 3, 5, 8 };
	std::vector<double> alpha_set{ 0.01, 0.02, 0.03, 0.04, 0.06, 0.08, 0.1, 0.15, 0.2, 0.3, 0.4, 0.6, 0.8, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0, 6.0, 8.0, 10.0 };
	std::vector<double> sigma_set{ 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2 };

	//std::vector<double> p_set{ 0.0, 0.2, 0.5, 0.8, 1.0 };
	//std::vector<double> bias_set{ 0, -1 };
	//std::vector<double> alpha_set{ 0.03, 0.1, 0.3, 1.0, 3.0, 10.0 };
	//std::vector<double> sigma_set{ 0.1, 0.2, 0.3, 0.5, 0.7, 0.9, 1.0 };

	for (int seed = 0; seed < loop; seed++) {
		for (auto nonlinear : nonlinear_vec) {
			for (auto p : p_set) {
				for (auto bias : bias_set) {
					for (auto alpha : alpha_set) {
						for (auto sigma : sigma_set) {
							for (auto toporogy_type : toporogy_types) {
								int connection_degree = unit_size / 10;
								if (toporogy_type == "ring") connection_degree = 1;
								if (toporogy_type == "doubly_ring") connection_degree = 2;
								if (toporogy_type == "sparse_random") connection_degree = 1;
								ret.push_back(reservoir_layer(unit_size, connection_degree, alpha, sigma, bias, p, nonlinear, seed, wash_out, toporogy_type));
							}
						}
					}
				}
			}
		}
	}
	std::mt19937 mt;
	std::shuffle(ret.begin(), ret.end(), mt);
	std::cerr << ret.size() << std::endl;
	return ret;
}

// 結合トポロジーや結合重みなどを設定する
void reservoir_layer::generate_reservoir() {

	std::uniform_real_distribution<> rand_minus1toplus1(-1, 1);
	std::uniform_int_distribution<> rand_0or1(0, 1);

	std::vector<int> permutation(unit_size);
	std::iota(permutation.begin(), permutation.end(), 1);
	//リザーバー層の結合をランダムに生成
	for (int n = 1; n <= unit_size; n++) {
		if (toporogy_type == "random") {
			std::shuffle(permutation.begin(), permutation.end(), mt);
			for (int k = 1; k <= connection_degree; k++) {
				adjacency_list[n][k] = permutation[k - 1];
			}
		}else if (toporogy_type == "sparse_random") {
			std::shuffle(permutation.begin(), permutation.end(), mt);
			if (n % 2 == 0) {
				adjacency_list[n].resize(connection_degree);
			}
			for (int k = 1; k < adjacency_list[n].size(); k++) {
				adjacency_list[n][k] = permutation[k - 1];
			}
		}
		else if (toporogy_type == "ring") {
			adjacency_list[n][1] = n + 1;
			if (n == unit_size) adjacency_list[n][1] = 1;
		}
		else if (toporogy_type == "doubly_ring") {
			adjacency_list[n][1] = n + 1;
			if (n == unit_size) adjacency_list[n][1] = 1;
			adjacency_list[n][2] = n - 1;
			if (n == 1) adjacency_list[n][2] = unit_size;
		}
		else {
			std::cerr << "no found toporogy_type:" << toporogy_type << std::endl;
		}
	}
	//各ノードが線形か非線形かを決定
	for (int n = 1; n <= unit_size; n++) {
		if (permutation[n - 1] <= unit_size * p) {
			node_type[n] = NON_LINEAR;
		}
		else
			node_type[n] = LINEAR;
	}
	for (int n = 1; n <= unit_size; n++) {
		//リザーバー層の結合重みを決定
		weight_reservoir[n][0] = rand_minus1toplus1(mt);
		for (int k = 1; k <= connection_degree; k++)
			weight_reservoir[n][k] = weight_factor * (1.0 / sqrt(connection_degree)) * (rand_0or1(mt) * 2 - 1);

		// 入力層の結合重みを決定
		input_signal_strength[n] = weight_factor * input_signal_factor * (rand_0or1(mt) * 2 - 1);
	}
}
/** リザーバー層を時間発展させる
	 * input_signal 入力信号
	 * output_node[t][n] 時刻tにおけるn番目のノードの出力
	 * t_size ステップ数
	 **/
void reservoir_layer::reservoir_update(const std::vector<double>& input_signal, std::vector<std::vector<double>>& output_node, const int t_size, int seed) {
	std::mt19937 mt2;
	mt2.seed(seed);
	std::uniform_real_distribution<> rand_minus1toplus1(-1, 1);
	output_node[0][0] = 1.0;
	for (int n = 1; n <= unit_size; n++) output_node[0][n] = rand_minus1toplus1(mt2);
	std::vector<double> input_sum_node(unit_size + 1, 0);
	for (int t = 0; t < t_size; t++) {
		for (int n = 1; n <= unit_size; n++) {
			input_sum_node[n] = input_signal_strength[n] * input_signal[t];
			for (int k = 1; k < adjacency_list[n].size(); k++) {
				input_sum_node[n] += weight_reservoir[n][k] * output_node[t][adjacency_list[n][k]];
			}
			input_sum_node[n] += weight_reservoir[n][0] * output_node[t][0] * bias_factor;
		}
		for (int n = 1; n <= unit_size; n++) {
			output_node[t + 1][n] = activation_function(input_sum_node[n], node_type[n]);
		}
		output_node[t + 1][0] = 1.0;
	}
}


// リザーバーの入出力を書き出す(ToDo:関数を一つに統合する）
void reservoir_layer::reservoir_update_show(const std::vector<double> input_signal, std::vector<std::vector<double>> output_node, const int t_size, const int wash_out, const std::string name) {

	std::uniform_real_distribution<> rand_minus1toplus1(-1, 1);
	output_node[0][0] = 1.0;
	for (int n = 1; n <= unit_size; n++) output_node[0][n] = rand_minus1toplus1(mt);
	std::ofstream outputfile("output_unit/" + name + ".txt");
	outputfile << "t,unit,input,output" << std::endl;
	std::vector<double> input_sum_node(unit_size + 1, 0);
	for (int t = 0; t <= t_size; t++) {
		for (int n = 1; n <= unit_size; n++) {
			input_sum_node[n] = input_signal_strength[n] * input_signal[t];
			for (int k = 1; k <= connection_degree; k++) {
				input_sum_node[n] += weight_reservoir[n][k] * output_node[t][adjacency_list[n][k]];
			}
			input_sum_node[n] += weight_reservoir[n][0] * output_node[t][0] * bias_factor;
		}
		output_node[t + 1][0] = 1.0;

		for (int n = 1; n <= unit_size; n++) {
			output_node[t + 1][n] = activation_function(input_sum_node[n], node_type[n]);
			if (t >= wash_out && t < wash_out + 200)
				outputfile << t << "," << n << "," << input_sum_node[n] << "," << output_node[t + 1][n] << std::endl;
				
		}
	}
	outputfile.close();
}
// Echo State Property(ESP)の有無をチェックする
// Echo State Propertyを持つリザーバーであるとは、リザーバーの持つノードの初期値に依存しない状態を言う。
//
bool reservoir_layer::calc_echo_state_property(const std::vector<double>& input_signal) {
	auto output_node1 = std::vector<std::vector<double>>(wash_out + 2, std::vector<double>(unit_size + 1, 0));
	auto output_node2 = std::vector<std::vector<double>>(wash_out + 2, std::vector<double>(unit_size + 1, 0));

	reservoir_update(input_signal, output_node1, wash_out, 1);
	reservoir_update(input_signal, output_node2, wash_out, 2);

	double err_sum = 0.0;
	for (int t = wash_out - 99; t <= wash_out; t++) {
		for (int n = 1; n <= unit_size; n++) {
			double err = (output_node1[t][n] - output_node2[t][n]);
			if (std::max(abs(output_node1[t][n]), abs(output_node2[t][n])) > 1000) {
				err += 100000.0;
				//std::cerr << output_node1[t][n] << "," << output_node2[t][n] << "," << weight_factor << std::endl;
				err_sum += err * err;
				break;
			}
			err_sum += err * err;
		}
	}
	// ノード初期値によって状態が等しくなるならば、EchoStatePropertyを持つ
	double err_ave = err_sum / (unit_size);
	//std::cerr << err_sum << std::endl;
	return is_echo_state_property = (err_ave <= 10.0);
}

double reservoir_layer::activation_function(const double x, const int type) {
	if (type == LINEAR) {
		return std::max(-10000.0, std::min(10000.0, x));
	}
	else if (type == NON_LINEAR) {
		return nonlinear(x);
	}
	assert(type != LINEAR && type != NON_LINEAR);
	return -1.0;
}