#include <string>
#include <fstream>
#include <sstream>
#include <iostream>
#include <map>
#include <vector>
#include <cmath>
#include <numeric>
#include <algorithm>
using namespace std;

const double max_L = 100;
const double max_NL = 500;
const int splitL_C = 50;
const int splitNL_C = 50;
const string NLtype = "NL_old";
const double split_L = max_L/splitL_C;
const double split_NL = max_NL/splitNL_C;
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
                cerr << str_conma_buf << endl;
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
int main (void) {
    string root_path = "../output_data/";
    vector<string> pathes{"NL_0_0.0_100_random"};
    for(auto path: pathes) {
        vector<vector<string>> data = input_csv(root_path + path + ".csv");
        for(int r = 1; r < data.size(); r++) {
            vector<string> elements = data[r];
            vector<double>& nmse_task = nmse[task_name];
            vector<vector<double>>& nmse_vec_task = nmse_vec[task_name];
            int NLr = stod(elements[mp[NLtype]]) / split_NL;
            int Lr = stod(elements[mp["L"]]) / split_L;
            const double tmp_nmse = stod(elements[mp[task_name]]);
            if (tmp_nmse > 1.05) 
                continue;
            nmse_task[NLr * splitL_C + Lr] += tmp_nmse;
            nmse_vec_task[NLr * splitL_C + Lr].push_back(tmp_nmse);
        }
        std::ofstream ofs_csv_file(root_path + path + "_add.csv");
        cerr << root_path + path + "_average.csv" << endl;
        vector<vector<string>> data = input_csv(root_path + path + ".csv");
        for(int r = 1; r < data.size(); r++) {
            
        }
        for(int i = 0; i < splitL_C; i++) {
            for(int j = 0; j < splitNL_C; j++) {
                int idx = i * splitL_C + j;
                if(nmse_cnt[idx]) {
                    ofs_csv_file << split_L * j << "," 
                                    << split_NL * i << "," 
                                    << nmse[task_name][idx] / nmse_cnt[idx] << "," 
                                    << nmse_cnt[idx] << "," << calc_SD(nmse_vec[task_name][idx]) << endl;
                }
            }
        }
    }
}