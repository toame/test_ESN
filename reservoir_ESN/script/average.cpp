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
map<string, pair<double, double>> component;
vector<vector<string>> input_csv(string path) {
    string str_buf, str_conma_buf;
    ifstream ifs_csv_file(path);
    vector<vector<string>> ret;
    int columns = 0, rows = 0;
    std::cerr << path << std::endl;
    while (getline(ifs_csv_file, str_buf)) {
        istringstream i_stream(str_buf);
        vector<string> elements;
        columns = 0;
        while (getline(i_stream, str_conma_buf, ',')) {
            elements.push_back(str_conma_buf);
            columns++;
        }
        rows++;
        if(elements.size() > 100) elements.resize(elements.size() + 2);
        ret.push_back(elements);
        if (rows % 100 == 0) cerr << rows << endl;
        if(rows > 50000) break;
    }
    return ret;
}
double calc_SD(std::vector<double>& data)
{
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
            if(elements[c].length() > 0) {
                tmp.push_back(stod(elements[c]));
            }
        }
        d_vec["NL_" + to_string(r)] = tmp;
    }
    std::cerr << "d_vec_size:" << d_vec.size() << std::endl;
}
void read_component(string path) {
    vector<vector<string>> data_vec = input_csv(path + "component.csv");
    std::cerr << data_vec.size() << std::endl;
    for(int r = 0; r < data_vec.size(); r++) {
        vector<string> elements = data_vec[r];
        vector<double> tmp;
        component[elements[0]] = make_pair(stod(elements[1]), stod(elements[2]));
        cerr << elements[0] << " " << component[elements[0]].first << " " << component[elements[0]].second << endl;
    }
    return;
}
vector<vector<string>> r_data;
int calc_L(string type, vector<string> elements) {
    if (mp.find(type) != mp.end()) {
        max_L = 100;
        split_L = max_L/splitL_C;
        return stod(elements[mp[type]]) / split_L;
    }
    assert(false);
    return 1;
}
int calc_NL(string type, vector<string> elements) {
    if (mp.find(type) != mp.end()) {
        if(type == "NL_old" || type == "NL_test" || type == "NL_test") max_NL = 500;
        if(type == "NL_old_2") max_NL = 200;
        split_NL = max_NL/splitNL_C;
        return stod(elements[mp[type]]) / split_NL;
    }
    assert(false);
    return 1;
}
string root_path = "../output_data/";
vector<string> pathes{"NL100"};
vector<string> task{"approx_3_0.0", "approx_6_-0.5", "approx_11_-1.0"};
void add_NL() {
    r_data[0][r_data[0].size() - 2] = "L_test";
    r_data[0][r_data[0].size() - 1] = "NL_test";
    std::ofstream ofs_csv_file(root_path + "add.csv");
    for(int i = 0; i < r_data[0].size();i++) {
        if (i != 0)  ofs_csv_file << ",";
        ofs_csv_file << r_data[0][i];
    }
    ofs_csv_file << endl;
    for(int i = 0; i < r_data[0].size(); i++) {
        mp[r_data[0][i]] = i;
    }
    for(int r = 1; r < r_data.size(); r++) {
        vector<string> elements = r_data[r];
        double L = 0.0;
        double NL = 0.0;
        int tau_0 = 0;
        int tau_1 = 1000;
        for (int i = 1; i <= 100; i++) {
            //if(stod(elements[mp["L_" + to_string(i)]]) >= 0.9) {
            tau_0 = max(i, tau_0);
            // L += stod(elements[mp["L_" + to_string(i)]]) * component["L_" + to_string(i)].first;
            // NL += stod(elements[mp["L_" + to_string(i)]]) * component["L_" + to_string(i)].second;
            // L += sqrt(i) * stod(elements[mp["L_" + to_string(i)]]);
            //}
            // if(stod(elements[mp["L_" + to_string(i)]]) >= 0.1) {
            //     L += stod(elements[mp["L_" + to_string(i)]]);
            // }
        }
        
        for(int i = 0; i < d_vec.size(); i++) {
            string task_name = "NL_" + to_string(i);
            //std::cerr << task_name << std::endl;
            vector<double> d = d_vec[task_name];
            int d_sum = 0;
            int tau = 0;
            int kind = 0;
            for(int i = 0; i < d.size(); i++) {
                d_sum += d[i];
                if (d[i] > 0) {
                    kind++;
                    tau = max(i, tau);
                }
            }
            double tmp_NL = stod(elements[mp[task_name]]);
            //if(tmp_NL < 0.1) continue;
            // L += sqrt(tau) * max(0.0, tmp_NL);
            // NL += sqrt(d_sum - 1.0) * max(0.0, tmp_NL);
                
            // if (tau > 0 && d_sum <= 6)
            //     NL += d_sum * max(0.0, tmp_NL);
            // L += tmp_NL * component[task_name].first;
            // NL += tmp_NL * component[task_name].second;
            if(d_sum <= 2 && kind == d_sum)
                L += d_sum * max(0.0, tmp_NL);
            else
                NL += d_sum * max(0.0, tmp_NL);
        }
        r_data[r].back() = (to_string(NL));
        for(int i = 0; i < elements.size() - 2;i++) {
            ofs_csv_file << elements[i] << ",";
        }
        ofs_csv_file << L << "," << NL << std::endl;
    }
    return;
}
int main (void) {
    
    read_d_vec(root_path);
    read_component(root_path);
    for(auto path: pathes) {
        r_data = input_csv(root_path + path + ".csv");
        cerr << "read completed" << endl;
        add_NL();
        
        cerr << "add completed" << endl;
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
            
            for(int r = 1; r < r_data.size(); r++) {
                vector<string> elements = r_data[r];
                int NLr = calc_NL(NL_type, elements);
                int Lr = calc_L(L_type, elements);
                //std::cerr << r << "," << NLr << "," << Lr << std::endl;
                nmse_cnt[NLr * splitL_C + Lr]++;
            }
            
            for(int r = 1; r < r_data.size(); r++) {
                vector<string> elements = r_data[r];
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