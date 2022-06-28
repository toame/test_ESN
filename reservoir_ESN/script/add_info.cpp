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

map<string, int> mp;

map<string, vector<double>> d_vec;
vector<string> get_line(string str_buf) {
    string str_conma_buf;
    istringstream i_stream(str_buf);
    vector<string> elements;
    while (getline(i_stream, str_conma_buf, ',')) {
        elements.push_back(str_conma_buf);
    }
    return elements;
}

vector<vector<string>> input_csv(string path) {
    string str_buf;
    ifstream ifs_csv_file(path);
    vector<vector<string>> ret;
    int columns = 0, rows = 0;
    std::cerr << path << std::endl;
    while (getline(ifs_csv_file, str_buf)) {
        vector<string> elements = get_line(str_buf);
        rows++;
        if(elements.size() > 100) elements.resize(elements.size() + 2);
        cerr << rows << " " << elements.size() << endl;
    }
    return ret;
}



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
vector<vector<string>> r_data;

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
        double L = 0.0;
        double NL = 0.0;
        vector<string> elements = r_data[r];
        int tau_0 = 0;
        for (int i = 1; i <= 10; i++) {
            if(stod(elements[mp["L_" + to_string(i)]]) >= 0.9) tau_0 = max(i, tau_0);
        }
        
        for(int i = 0; i < d_vec.size(); i++) {
            string task_name = "NL_" + to_string(i);
            //std::cerr << task_name << std::endl;
            vector<double> d = d_vec[task_name];
            int d_sum = 0;
            int tau = 0;
            int max_d = 0;
            for(int i = 0; i < d.size(); i++) {
                d_sum += d[i];
                max_d = max<int>(max_d, d[i]);
                if (d[i] > 0) {
                    tau = max(i, tau);
                }
            }
            double tmp_NL = stod(elements[mp[task_name]]);

            //NL += d_sum/(double)(tau + 1.0) * max(0.0, tmp_NL);
            NL += d_sum * max(0.0, tmp_NL);
            if(d_sum == 2) {
                //L += tau/(double)(tau + d_sum - 1.0) * max(0.0, tmp_NL);
                L += d_sum * max(0.0, tmp_NL);
            }
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
    for(auto path: pathes) {
        input_csv(root_path + path + ".csv");
    }
}