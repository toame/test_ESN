#include <string>
#include <fstream>
#include <sstream>
#include <iostream>
#include <map>
#include <vector>
using namespace std;

const double max_L = 200;
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

int main (void) {
    string root_path = "../output_data/";
    vector<string> pathes{"NL_0_0.0_50_random"};
    for(auto path: pathes) {
        vector<vector<string>> data = input_csv(root_path + path + ".csv");
        vector<double> nmse1(splitL_C * splitL_C), nmse2(splitL_C * splitL_C), nmse3(splitL_C * splitL_C);
        vector<int> nmse_cnt(splitL_C * splitL_C);
        for(int r = 1; r < data.size(); r++) {
            vector<string> elements = data[r];
            int NLr = stod(elements[mp[NLtype]]) / split_NL;
            int Lr = stod(elements[mp["L"]]) / split_L;
            const double tmp_nmse1 = stod(elements[mp["approx_3_0.0"]]);
            const double tmp_nmse2 = stod(elements[mp["approx_6_-0.5"]]);
            const double tmp_nmse3 = stod(elements[mp["approx_11_-1.0"]]);
            //cerr << tmp_nmse1 << "," << tmp_nmse2 << "," << tmp_nmse3 << endl;
            if (tmp_nmse1 > 1.05 || tmp_nmse2 > 1.05 || tmp_nmse3 > 1.05) 
                continue;
            nmse1[NLr * splitL_C + Lr] += tmp_nmse1;
            nmse2[NLr * splitL_C + Lr] += tmp_nmse2;
            nmse3[NLr * splitL_C + Lr] += tmp_nmse3;
            nmse_cnt[NLr * splitL_C + Lr]++;
        }
        vector<string> task{"approx_3_0.0", "approx_6_-0.5", "approx_11_-1.0"};
        for(int id = 0; id < 3; id++) {
            std::ofstream ofs_csv_file(root_path + path + "_" + task[id] + "_" + NLtype + "_average.csv");
            cerr << root_path + path + "_average.csv" << endl;
            ofs_csv_file << "L,NL,nmse" << endl;
            for(int i = 0; i < splitL_C; i++) {
                for(int j = 0; j < splitNL_C; j++) {
                    int idx = i * splitL_C + j;
                    if(nmse_cnt[idx]) {
                        if(id == 0)
                            ofs_csv_file << split_L * j << "," << split_NL * i << "," <<nmse1[idx]/nmse_cnt[idx] << endl;
                        if(id == 1)
                            ofs_csv_file << split_L * j << "," << split_NL * i << "," <<nmse2[idx]/nmse_cnt[idx] << endl;
                        if(id == 2)
                            ofs_csv_file << split_L * j << "," << split_NL * i << "," <<nmse3[idx]/nmse_cnt[idx] << endl;
                    }
                }
            }
        }
    }

}