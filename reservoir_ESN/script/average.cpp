#include <string>
#include <fstream>
#include <sstream>
#include <iostream>
#include <map>
#include <vector>
#include <cmath>
#include <numeric>
#include <algorithm>
#include <cassert>
using namespace std;

const int splitL_C = 50;
const int splitNL_C = 50;

map<string, int> mp;
vector<vector<string>> input_csv(string path) {
    string str_buf, str_conma_buf;
    ifstream ifs_csv_file(path);
    vector<vector<string>> ret;
    int columns = 0, rows = 0;
    while (getline(ifs_csv_file, str_buf)) {
        istringstream i_stream(str_buf);
        vector<string> elements;
        while (getline(i_stream, str_conma_buf, ',')) {
            if (rows == 0) {
                mp[str_conma_buf] = columns;
            }
            else {
                elements.push_back(str_conma_buf);
            }
            columns++;
        }
        rows++;
        ret.push_back(elements);
    }
    return ret;
}
double calc_SD(std::vector<double>& data)
{
    // 1つ目の手法(accumulate版)　※yumetodoさんのアイディア
    // 第四引数に関数オブジェクトやラムダ式を渡せば加算以外の演算も可能
    // (下コードでは引数のsumが直前の結果・eがその段階で読み込んだ配列の要素)
    const auto ave = std::accumulate(std::begin(data), std::end(data), 0.0) / std::size(data);
    const auto var = std::accumulate(std::begin(data), std::end(data), 0.0, [ave](double sum, const auto& e){
        const auto temp = e - ave;
        return sum + temp * temp;
    }) / std::size(data);
    return std::sqrt(var);
}
double max_L, max_NL;
double split_L, split_NL;
map<string, vector<double>> d_vec;
void read_d_vec(string path) {
    vector<vector<string>> data_vec = input_csv(path + "d_vec.csv");
    for(int r = 0; r < data_vec.size(); r++) {
        vector<string> elements = data_vec[r];
        vector<double> tmp;
        for(int c = 0; c < elements.size(); c++) {
            tmp.push_back(stod(elements[c]));
        }
        d_vec["NL_" + to_string(r)] = tmp;
    }
    std::cerr << "d_vec_size:" << d_vec.size() << std::endl;
}
int calc_L(string type, vector<string> elements) {
    if (mp.find(type) != mp.end()) {
        max_L = 100;
        split_L = max_L/splitL_C;
        return stod(elements[mp[type]]) / split_L;
    } else {
        assert(false);
    }
}
int calc_NL(string type, vector<string> elements) {
    if (mp.find(type) != mp.end()) {
        if(type == "NL_old") max_NL = 500;
        if(type == "NL_old_2") max_NL = 200;
        split_NL = max_NL/splitNL_C;
        return stod(elements[mp[type]]) / split_NL;
    } else if(type == "NL_test") {
        max_NL = 500;
        split_NL = max_NL/splitNL_C;
        double NL = 0;
        for(int i = 0; i < d_vec.size(); i++) {
            string task_name = "NL_" + to_string(i);
            //std::cerr << task_name << std::endl;
            vector<double>d = d_vec[task_name];
            int d_sum = 0;
            for(int i = 0; i < d.size(); i++) {
                d_sum += d[i];
            }
            double tmp_NL = stod(elements[mp[task_name]]);
            if(tmp_NL > 0.01)
                NL += d_sum * max(0.0, tmp_NL);
        }
        return NL / split_NL;
    } else {
        assert(false);
    }
}
int main (void) {
    string root_path = "../output_data/";
    vector<string> pathes{"NL_0_0.0_100_random"};
    vector<string> task{"approx_3_0.0", "approx_6_-0.5", "approx_11_-1.0"};
    read_d_vec(root_path);
    for(auto path: pathes) {
        vector<vector<string>> data = input_csv(root_path + path + ".csv");
        cerr << "read completed" << endl;
        vector<string> NL_type_vec = {"NL_test"};
        string L_type = "L";
        for(auto NL_type: NL_type_vec) {
            std::cerr << NL_type << std::endl;
            map<string, vector<double>> nmse;
            map<string, vector<vector<double>>> nmse_vec;
            for(auto task_name: task) {
                nmse[task_name].resize((splitL_C * splitL_C));
                nmse_vec[task_name].resize(splitL_C * splitL_C);
            }
            vector<int> nmse_cnt(splitL_C * splitL_C);
            
            for(int r = 1; r < data.size(); r++) {
                vector<string> elements = data[r];
                int NLr = calc_NL(NL_type, elements);
                int Lr = calc_L(L_type, elements);
                //std::cerr << r << "," << NLr << "," << Lr << std::endl;
                nmse_cnt[NLr * splitL_C + Lr]++;
            }
            
            for(int r = 1; r < data.size(); r++) {
                vector<string> elements = data[r];
                int NLr = calc_NL(NL_type, elements);
                int Lr = calc_L(L_type, elements);
                //std::cerr << r << "," << NLr << "," << Lr << std::endl;
                for(auto task_name: task) {
                    vector<double>& nmse_task = nmse[task_name];
                    vector<vector<double>>& nmse_vec_task = nmse_vec[task_name];
                    const double tmp_nmse = stod(elements[mp[task_name]]);
                    if (tmp_nmse > 1.05) 
                        continue;
                    nmse_task[NLr * splitL_C + Lr] += tmp_nmse;
                    nmse_vec_task[NLr * splitL_C + Lr].push_back(tmp_nmse);
                }
            }
            
            std::ofstream ofs_csv_file(root_path + path + "_" + NL_type + "_average.csv");
            cerr << root_path + path + "_average.csv" << endl;
            ofs_csv_file << "L,NL";
            for(auto task_name: task) 
                ofs_csv_file << "," << task_name << "_nmse," << task_name << "_count," << task_name << "_SD";
            ofs_csv_file << endl;
            for(int i = 0; i < splitL_C; i++) {
                for(int j = 0; j < splitNL_C; j++) {
                    int idx = i * splitL_C + j;
                    if(nmse_cnt[idx] == 0) continue;
                    ofs_csv_file << split_L * j << "," << split_NL * i;
                    for(auto task_name: task) {
                        ofs_csv_file << "," << nmse[task_name][idx] / nmse_cnt[idx] << ","  << nmse_cnt[idx] << "," << calc_SD(nmse_vec[task_name][idx]);
                    }
                    ofs_csv_file << endl;
                }
            }
        }
    }
}