#include <cblas.h>
#include <vector>
class output_learning {
public:
    std::vector<std::vector<double>> A, L;
    std::vector<double> d, b, w;
    output_learning();

    //�A���ꎟ������Aw=b��A�𐶐�
    void generate_simultaneous_linear_equationsA(const std::vector<std::vector<double>>& output_node, const int wash_out, const int step, const int n_size);
   
    //�A���ꎟ������Aw=b��b�𐶐�
    void generate_simultaneous_linear_equationsb(const std::vector<std::vector<double>>& output_node, const std::vector<double>& yt_s, const int wash_out, const int step, const int n_size);

    //�A���ꎟ������Aw=b��b�𐶐�
    void generate_simultaneous_linear_equationsb_fast(const std::vector<double>& output_node, const std::vector<double>& yt_s, const int wash_out, const int step, const int n_size);


    // �A���ꎟ������Aw = b ��w�ɂ���ICCG�ŉ���
    // void Learning(std::vector<double>& w, const std::vector<std::vector<double>>& A, const std::vector<double>& b, const double lambda, const int n_size);

    // �s���S�R���X�L�[����
    int IncompleteCholeskyDecomp2(int n);
    

    // p_0 = (LDL^T)^-1 r_0 �̌v�Z
    void ICRes(const std::vector<double>& r, std::vector<double>& u, int n);

    // ���ς����߂�
    double dot(const std::vector<double> r1, const std::vector<double> r2, const int size);

    /*!
     * �s���S�R���X�L�[�����ɂ��O�����t�������z�@�ɂ��A�Ex=b������
     * @param[in] A n�~n���l�Ώ̍s��
     * @param[in] b �E�Ӄx�N�g��
     * @param[out] x ���ʃx�N�g��
     * @param[in] n �s��̑傫��
     * @param[inout] max_iter �ő唽����(�����I����,���ۂ̔�������Ԃ�)
     * @param[inout] eps ���e�덷(�����I����,���ۂ̌덷��Ԃ�)
     * @return 1:����,0:���s
     */
    int ICCGSolver(int n, int& max_iter, double& eps);

    /*!
     * �������z�@�ɂ��A�Ex=b������
     * @param[in] A n�~n���l�Ώ̍s��
     * @param[in] b �E�Ӄx�N�g��
     * @param[out] x ���ʃx�N�g��
     * @param[in] n �s��̑傫��
     * @param[inout] max_iter �ő唽����(�����I����,���ۂ̔�������Ԃ�)
     * @param[inout] eps ���e�덷(�����I����,���ۂ̌덷��Ԃ�)
     * @return 1:����,0:���s
     */
    int CGSolver(const std::vector<std::vector<double>>& A, const std::vector<double>& b, std::vector<double>& x, int n, int& max_iter, double& eps);
};

inline double output_learning::dot(const std::vector<double> r1, const std::vector<double> r2, const int size) {
    return cblas_ddot(size, r1.data(), 1, r2.data(), 1);
}

// p_0 = (LDL^T)^-1 r_0 �̌v�Z
inline void output_learning::ICRes(const std::vector<double>& r, std::vector<double>& u, int n) {
    std::vector<double> y(n);
    for (int i = 0; i < n; ++i) {
        double rly = r[i];
        for (int j = 0; j < i; ++j) {
            rly -= L[i][j] * y[j];
        }
        y[i] = rly / L[i][i];
    }

    for (int i = n - 1; i >= 0; --i) {
        double lu = 0.0;
        for (int j = i + 1; j < n; ++j) {
            lu += L[j][i] * u[j];
        }
        u[i] = y[i] - d[i] * lu;
    }
}
