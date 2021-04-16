#include <cblas.h>
#include <vector>
class output_learning {
public:
    std::vector<std::vector<double>> A;
    std::vector<double> w;
    output_learning();

    //連立一次方程式Aw=bのAを生成
    void generate_simultaneous_linear_equationsA(const std::vector<std::vector<double>>& output_node, const int wash_out, const int step, const int n_size);
   
    //連立一次方程式Aw=bのbを生成
    void generate_simultaneous_linear_equationsb(std::vector<double>& b, const std::vector<std::vector<double>>& output_node, const std::vector<double>& yt_s, const int wash_out, const int step, const int n_size);


    // 連立一次方程式Aw = b をwについてICCGで解く
    // void Learning(std::vector<double>& w, const std::vector<std::vector<double>>& A, const std::vector<double>& b, const double lambda, const int n_size);

    // 不完全コレスキー分解
    int IncompleteCholeskyDecomp2(std::vector<std::vector<double>>& L, std::vector<double>& d, int n);
    

    // p_0 = (LDL^T)^-1 r_0 の計算
    void ICRes(const std::vector<std::vector<double>>& L, const std::vector<double>& d, const std::vector<double>& r, std::vector<double>& u, int n);

    // 内積を求める
    double dot(const std::vector<double> r1, const std::vector<double> r2, const int size);

    /*!
     * 不完全コレスキー分解による前処理付共役勾配法によりA・x=bを解く
     * @param[in] A n×n正値対称行列
     * @param[in] b 右辺ベクトル
     * @param[out] x 結果ベクトル
     * @param[in] n 行列の大きさ
     * @param[inout] max_iter 最大反復数(反復終了後,実際の反復数を返す)
     * @param[inout] eps 許容誤差(反復終了後,実際の誤差を返す)
     * @return 1:成功,0:失敗
     */
    int ICCGSolver(const std::vector<std::vector<double>>& L, const std::vector<double>& d, const std::vector<double>& b, std::vector<double>& x, int n, int& max_iter, double& eps);

    /*!
     * 共役勾配法によりA・x=bを解く
     * @param[in] A n×n正値対称行列
     * @param[in] b 右辺ベクトル
     * @param[out] x 結果ベクトル
     * @param[in] n 行列の大きさ
     * @param[inout] max_iter 最大反復数(反復終了後,実際の反復数を返す)
     * @param[inout] eps 許容誤差(反復終了後,実際の誤差を返す)
     * @return 1:成功,0:失敗
     */
    int CGSolver(const std::vector<std::vector<double>>& A, const std::vector<double>& b, std::vector<double>& x, int n, int& max_iter, double& eps);
};

inline double output_learning::dot(const std::vector<double> r1, const std::vector<double> r2, const int size) {
    return cblas_ddot(size, r1.data(), 1, r2.data(), 1);
}
