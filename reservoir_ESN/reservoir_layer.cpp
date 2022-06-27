#include "reservoir_layer.h"
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
double TD_MG(const double x, double J, double input_gain, double feed_gain) {
	return (feed_gain * (x + input_gain * J)) / (1.0 + pow(x + input_gain * J, 4.0));//ρ = 1～10  //////////////変更要素/////////////////
}

double TD_ikeda(const double x, double J, double input_gain, double feed_gain) {
	return feed_gain * pow(sin(x + input_gain * J + 0.5), 2.0);//Φ=0.2～0.7(0.05刻み)  //////////////変更要素/////////////////
}

double TD_exp(const double x, double J, double input_gain, double feed_gain) {
	return feed_gain * exp(-x) * sin(x + input_gain * J);
}
reservoir_layer::reservoir_layer() {}
reservoir_layer::reservoir_layer(const int unit_size, const int connection_degree,const double iss_factor, const double input_gain, const double feed_gain, const double p,
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
	this->input_gain = input_gain;
	this->feed_gain = feed_gain;
	if (nonlinear_name == "TD_ikeda") nonlinear = TD_ikeda;
	else if (nonlinear_name == "TD_exp") nonlinear = TD_exp;
	else assert(false);
	if (bias_factor < -0.9) this->bias_factor = input_signal_factor * weight_factor;
	node_type.resize(unit_size + 1);
	adjacency_list.resize(unit_size + 1, std::vector<int>(connection_degree + 1));
	weight_reservoir.resize(unit_size + 1, std::vector<double>(connection_degree + 1));
	J.resize(4000 + 1, std::vector<double>(unit_size + 1));
	input_signal_strength.resize(unit_size + 1);
	mt.seed(seed);
}

std::vector<reservoir_layer> reservoir_layer::generate_reservoir(const int unit_size, const unsigned int loop, const int wash_out) {
	std::vector<reservoir_layer> ret;


	std::vector<std::string> nonlinear_vec{ "TD_ikeda", "TD_exp" };
	std::vector<std::string> toporogy_types{ "random" };
	//std::vector<double> p_set{ 0.05, 0.1, 0.2, 0.35, 0.5, 0.65, 0.8, 0.9, 0.95, 1.0, 0.0 };
	//std::vector<double> bias_set{ 0, 1, 2, 3, 5, 8 };
	//std::vector<double> alpha_set{ 0.01, 0.02, 0.03, 0.04, 0.06, 0.08, 0.1, 0.15, 0.2, 0.3, 0.4, 0.6, 0.8, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0, 6.0, 8.0, 10.0 };
	//std::vector<double> sigma_set{ 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2 };

	std::vector<double> p_set{ 0.2, 0.5, 0.8, 1.0 };
	std::vector<double> bias_set{ 0.1, 0.15, 0.2, 0.25, 0.3 };
	std::vector<double> alpha_set{ 0.1, 0.2, 0.4, 0.8, 1.6, 3.2, 6.4, 12.8 };
	std::vector<double> sigma_set{ 0.7, 0.8, 0.9, 1.0, 1.1, 1.2 };

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

	a = { -1.0, -0.6, -0.2, 0.2, 0.6, 1.0 };
	b = { -1.0,1.0 };

	std::vector<int> permutation(unit_size + 1);
	std::iota(permutation.begin(), permutation.end(), 1);

	for (int n = 1; n <= unit_size; n++) {
		std::shuffle(permutation.begin(), permutation.end(), mt);
	}


	//各ノードが線形か非線形かを決定
	for (int n = 1; n <= unit_size; n++) {
		if (permutation[n] <= unit_size * p) {
			node_type[n] = NON_LINEAR;
		}
		else
			node_type[n] = LINEAR;
	}

	//マスク信号と入力の強みをここで一緒にしている
	for (int n = 1; n <= unit_size; n++) {
		//input_signal_strength[n] = input_signal_factor * (double)(rand_minus1orplus1(mt) / 2.0);
		input_signal_strength[n] = input_signal_factor * a[rand() % a.size()];
		//input_signal_strength[n] = input_signal_factor * b[rand() % b.size()];///////////////////////////変更要素//////////////////////
	}
}
/** リザーバー層を時間発展させる
	 * input_signal 入力信号
	 * output_node[t][n] 時刻tにおけるn番目のノードの出力
	 * t_size ステップ数
	 **/
void reservoir_layer::reservoir_update(const std::vector<double>& input_signal, std::vector<std::vector<double>>& output_node, const int t_size, int seed) {
	//std::mt19937 mt2;
	//mt2.seed(seed);
	//std::uniform_real_distribution<> rand_minus1toplus1(-1, 1);
	//output_node[0][0] = 1.0;
	//for (int n = 1; n <= unit_size; n++) output_node[0][n] = rand_minus1toplus1(mt2);
	//std::vector<double> input_sum_node(unit_size + 1, 0);
	//for (int t = 0; t < t_size; t++) {
	//	for (int n = 1; n <= unit_size; n++) {
	//		input_sum_node[n] = input_signal_strength[n] * input_signal[t];
	//		for (int k = 1; k < adjacency_list[n].size(); k++) {
	//			input_sum_node[n] += weight_reservoir[n][k] * output_node[t][adjacency_list[n][k]];
	//		}
	//		input_sum_node[n] += weight_reservoir[n][0] * output_node[t][0] * bias_factor;
	//	}
	//	for (int n = 1; n <= unit_size; n++) {
	//		output_node[t + 1][n] = activation_function(input_sum_node[n], node_type[n]);
	//	}
	//	output_node[t + 1][0] = 1.0;
	//}
	std::mt19937 mt2; // メルセンヌ・ツイスタの32ビット版
	mt2.seed(seed);
	std::uniform_real_distribution<> rand_minus1toplus1(-1, 1);
	double exp(double x);
	output_node[0][0] = 1.0;
	for (int n = 1; n <= unit_size; n++) output_node[0][n] = (double)rand_minus1toplus1(mt2);

	double ξ, d;
	d = 2.0 / (double)unit_size;/////////////////////////////////////////////////////////変更要素////////////////////////////////////////////////////////////////////
	ξ = log(1.0 + d);

	//マスク信号を加えた最終的な入力信号
	for (int t = 1; t <= t_size; t++) {
		for (int n = 1; n <= unit_size; n++) {
			J[t][n] = input_signal[t - 1] * input_signal_strength[n];
		}
	}

	//初期値設定
	for (int n = 1; n <= unit_size; n++) {
		J[0][n] = input_signal_strength[n];
		output_node[0][n] = activation_function(output_node[0][n], 0.0, node_type[n], J[0][n]);

		//output_node[0][n] *= d / (1.0 + d); /////////////////////////変更要素//////////////////////
		output_node[0][n] *= (1.0 - exp(-ξ));///////////////////////////////////////////////////////

		//output_node[0][n] += (1.0 / (1.0 + d)) * (output_node[0][n - 1]);//////////////////////変更要素////////////////////
		output_node[0][n] += exp(-ξ) * (output_node[0][n - 1]);/////////////////////////////////////////////////////////////
	}

	//通常の時間遅延システム型時間発展式
	for (int t = 1; t <= t_size; t++) {
		output_node[t][0] = output_node[t - 1][unit_size];
		for (int n = 1; n <= unit_size; n++) {
			output_node[t][n] = activation_function(output_node[t - 1][n], 0.0, node_type[n], J[t][n]);

			//output_node[t][n] *= (d / (1.0 + d));//////////////変更要素//////////////
			output_node[t][n] *= (1.0 - exp(-ξ));/////////////////////////////////////

			//output_node[t][n] += (1.0 / (1.0 + d)) * (output_node[t][n - 1]);//////////変更要素////////////
			output_node[t][n] += exp(-ξ) * (output_node[t][n - 1]);/////////////////////////////////////////
		}
	}
}


//// リザーバーの入出力を書き出す(ToDo:関数を一つに統合する）
//void reservoir_layer::reservoir_update_show(const std::vector<double> input_signal, std::vector<std::vector<double>> output_node, const int t_size, const int wash_out, const std::string name) {
//
//	std::uniform_real_distribution<> rand_minus1toplus1(-1, 1);
//	output_node[0][0] = 1.0;
//	for (int n = 1; n <= unit_size; n++) output_node[0][n] = rand_minus1toplus1(mt);
//	std::ofstream outputfile("output_unit/" + name + ".txt");
//	outputfile << "t,unit,input,output" << std::endl;
//	std::vector<double> input_sum_node(unit_size + 1, 0);
//	for (int t = 0; t <= t_size; t++) {
//		for (int n = 1; n <= unit_size; n++) {
//			input_sum_node[n] = input_signal_strength[n] * input_signal[t];
//			for (int k = 1; k <= connection_degree; k++) {
//				input_sum_node[n] += weight_reservoir[n][k] * output_node[t][adjacency_list[n][k]];
//			}
//			input_sum_node[n] += weight_reservoir[n][0] * output_node[t][0] * bias_factor;
//		}
//		output_node[t + 1][0] = 1.0;
//
//		for (int n = 1; n <= unit_size; n++) {
//			output_node[t + 1][n] = activation_function(input_sum_node[n], node_type[n]);
//			if (t >= wash_out && t < wash_out + 200)
//				outputfile << t << "," << n << "," << input_sum_node[n] << "," << output_node[t + 1][n] << std::endl;
//
//		}
//	}
//	outputfile.close();
//}
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

double reservoir_layer::activation_function(const double x1, const double x2, const int type, const double J) {
	double x;
	x = (x1 + x2);
	//x = (x1 + x2)/2

	if (type == LINEAR) {
		//return feed_gain * (std::max(-1000.0, std::min(1000.0, x)) + input_gain * J); 
		return std::max(-1000.0, std::min(1000.0, x));
	}
	else if (type == NON_LINEAR) {
		return nonlinear(x, J, input_gain, feed_gain);
	}
	assert(type != LINEAR && type != NON_LINEAR);
	return -1.0;
}