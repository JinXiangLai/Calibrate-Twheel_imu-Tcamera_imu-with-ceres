#include "common.h"


using namespace std;

#define USE_AUTO_DIFF 0

#if USE_AUTO_DIFF
// 定义误差函数，自动求导条件下，不需定义Evaluate函数，但必须提供operator()函数以计算残差
struct RotationResidual {
    RotationResidual(const Eigen::Vector3d& v_bi, const Eigen::Vector3d& v_li)
        : v_bi_(v_bi), v_li_(v_li) {}

    // 使用自动求导时才必须提供该函数
    template <typename T>
    bool operator()(const T* const q, T* residual) const {
        // 将四元数转换为 Eigen::Quaternion
        Eigen::Quaternion<T> q_rot(q[0], q[1], q[2], q[3]);

        // 将轮速计速度转换为 Eigen::Vector3
        Eigen::Matrix<T, 3, 1> v_li_T = v_li_.cast<T>();

        // 计算旋转后的速度
        Eigen::Matrix<T, 3, 1> predicted_v_bi = q_rot * v_li_T;

        // 计算残差
        residual[0] = predicted_v_bi(0) - T(v_bi_(0));
        residual[1] = predicted_v_bi(1) - T(v_bi_(1));
        residual[2] = predicted_v_bi(2) - T(v_bi_(2));

        return true;
    }

   private:
    const Eigen::Vector3d v_bi_;  // IMU 速度
    const Eigen::Vector3d v_li_;  // 轮速计速度
};

#else

// 第2个参数对应Manifold的环境维度还是切空间维度呢？
class RotationResidual : public ceres::CostFunction {
   public:
    RotationResidual(const Eigen::Vector3d& v_bi, const Eigen::Vector3d& v_li)
        : v_bi_(v_bi), v_li_(v_li) {
        // SizedCostFunction同样继承自CostFunction，
        // 若不使用SizedCostFunction，则需要手动设置残差维度和参数块大小
        set_num_residuals(3);
        // 这里
        mutable_parameter_block_sizes()->push_back(4);
    }

    // 使用解析雅可比需要提供Evaluate函数，此时不需要提供operator()函数
    // 所以这里是自动求导和解析雅可比的区别
    virtual bool Evaluate(double const* const* parameters, double* residuals,
                          double** jacobians) const override {
        // 将四元数转换为 Eigen::Quaternion
        const double* q = parameters[0];
        Eigen::Quaternion<double> q_rot(q[0], q[1], q[2], q[3]);

        // 将轮速计速度转换为 Eigen::Vector3
        Eigen::Matrix<double, 3, 1> v_li_T = v_li_.cast<double>();

        // 计算旋转后的速度
        Eigen::Matrix<double, 3, 1> predicted_v_bi = q_rot * v_li_T;

        // 计算残差
        residuals[0] = predicted_v_bi(0) - v_bi_(0);
        residuals[1] = predicted_v_bi(1) - v_bi_(1);
        residuals[2] = predicted_v_bi(2) - v_bi_(2);

        // 手动提供解析雅可比矩阵
        // 对于Manifold，这里提供的是关于环境维度的雅可比矩阵
        // 所以不能直接使用李代数的结果来赋值，这里提供∂e/∂q，而不是∂e/∂δ
        // 由于四元数可以直接转为旋转矩阵，因此e=q*v=R_q*v，这样就可以得到e关于[qw,
        // qx, qy, qz]的雅可比矩阵了 但这种做法很明显比较复杂，不是我的首选

        // 这里需要首先判断jacobians是否为0x0，因为在TRUST_REGION策略下，我们调用Evaluate函数的目的可能只是需要计算residuals
        if (jacobians) {
            if (jacobians[0]) {
                // Evaluate函数中的雅可比是关于环境维度的，所以维度是[3x4]
                Eigen::Map<Eigen::Matrix<double, 3, 4, Eigen::RowMajor>> j(
                    jacobians[0]);
                // 右乘扰动更新雅可比
                // 这里给出了∂e/∂δ的李代数雅可比，因此要注意PlusJacobian中要求实现∂q/∂δ不再是真实值，
                // 而是要保证该处雅可比j右乘上它，矩阵值可以保持不变。
                // 此外需要注意，不能使用ceres::TRUST_REGION，只能使用ceres::LINE_SEARCH，
                // 因为PlusJacobian中的雅可比矩阵不是真实的，ceres2不能用其计算信任域
                j.setZero();
                j.block(0, 0, 3, 3) = -q_rot.toRotationMatrix() * skew(v_li_T);

                // 还是core dump
                // Eigen::Matrix<double, 3, 4> J_w_q;
                // J_w_q << -0.5 * q[1], -0.5 * q[2], -0.5 * q[3], 0.5 * q[0],
                //             0.5 * q[0], -0.5 * q[3],  0.5 * q[2], 0.5 * q[1],
                //             0.5 * q[3],  0.5 * q[0], -0.5 * q[1], 0.5 * q[2];
                // j.block(0, 0, 3, 4) = j.block(0, 0, 3, 3)*J_w_q;
            }
        }

        return true;
    }

   private:
    const Eigen::Vector3d v_bi_;  // IMU 速度
    const Eigen::Vector3d v_li_;  // 轮速计速度
};
#endif

// 自定义四元数的 LocalParameterization
class QuaternionParameterization : public ceres::Manifold {
   public:
    bool Plus(const double* x, const double* delta,
              double* x_plus_delta) const override {
        // 将 delta 转换为四元数
        Eigen::Matrix<double, 3, 1> delta_q_vec(delta[0], delta[1], delta[2]);
        double theta = delta_q_vec.norm();
        Eigen::Quaterniond delta_q;
        if (theta > 0.0) {
            delta_q = Eigen::Quaterniond(
                Eigen::AngleAxisd(theta, delta_q_vec / theta));
        } else {
            delta_q = Eigen::Quaterniond(1.0, 0.0, 0.0, 0.0);
        }

        // 将 x 转换为四元数
        Eigen::Quaterniond x_q(x[0], x[1], x[2], x[3]);

        // 更新四元数
        Eigen::Quaterniond x_plus_delta_q = x_q * delta_q;
        x_plus_delta_q.normalize();

        // 将结果写回
        x_plus_delta[0] = x_plus_delta_q.w();
        x_plus_delta[1] = x_plus_delta_q.x();
        x_plus_delta[2] = x_plus_delta_q.y();
        x_plus_delta[3] = x_plus_delta_q.z();

        return true;
    }

    virtual bool ComputeJacobian(const double* x, double* jacobian) const {
        return false;
    }

    virtual bool RightMultiplyByPlusJacobian(
        const double* x, const int num_rows, const double* ambient_matrix,
        double* tangent_matrix) const override {
        return false;
    }

    virtual bool Minus(const double* x, const double* y,
                       double* y_minus_x) const override {
        return false;
    }

    virtual bool PlusJacobian(const double* x,
                              double* jacobian) const override {
        // 计算的是Plus函数对delta增量的雅可比矩阵

#if USE_AUTO_DIFF
        Eigen::Map<Eigen::Matrix<double, 3, 4, Eigen::RowMajor>> j(jacobian);
        j.setZero();
        // j(1, 0) = 1.0;
        // j(2, 1) = 1.0;
        // j(3, 2) = 1.0;
        // 以下参考deepseek的推导，使用下面的公式，收敛明显加快，
        // 相同条件下，下面的雅可比只需迭代5次，
        // 上面的雅可比需要迭代48次
        // clang-format off
        j << -0.5 * x[1], -0.5 * x[2], -0.5 * x[3], 0.5 * x[0], 
            -0.5 * x[3], 0.5 * x[2], 0.5 * x[3], 0.5 * x[0], 
            -0.5 * x[1], -0.5 * x[2], 0.5 * x[1], 0.5 * x[0];
        // clang-format on
#else
        // 因为我在Evaluate函数中已经直接给出∂e/∂δ，所以这里只需要保证乘上∂e/∂δ值不变即可
        // 但此时只能使用ceres::LINE_SEARCH，而不能使用ceres::TRUST_REGION
        Eigen::Map<Eigen::Matrix<double, 4, 3, Eigen::RowMajor>> j(jacobian);
        j.setZero();
        j.block(0, 0, 3, 3).setIdentity();
#endif
        return true;
    }

    virtual bool MinusJacobian(const double* x,
                               double* jacobian) const override {
        return false;
    }

    int AmbientSize() const override { return 4; }  // 四元数的环境维度
    int TangentSize() const override { return 3; }  // 四元数的切空间维度
};

int main() {
    // 示例数据：IMU 和轮速计的速度测量值
    std::vector<Eigen::Vector3d> v_bis = {
        {1.0, 0.0, 0.0}, {0.0, 1.0, 0.0}, {0.0, 0.0, 1.0}};

    std::vector<Eigen::Vector3d> v_lis = {
        {0.9, 0.1, 0.0}, {0.1, 0.9, 0.0}, {0.0, 0.0, 1.1}};

    Eigen::Matrix3d Rbv =
        Eigen::AngleAxisd(0.13, Eigen::AngleAxisd::Vector3::UnitY()) *
        Eigen::AngleAxisd(1.3, Eigen::AngleAxisd::Vector3::UnitX())
            .toRotationMatrix();
    for (int i = 0; i < v_bis.size(); ++i) {
        v_bis[i] = Rbv * v_lis[i];
    }

    // 初始四元数（单位四元数）
    double q[4] = {0.5, 0.5, 0.7071, 0.0001};  // 初始化为无旋转

    // 构建优化问题
    ceres::Problem problem;
    // 1、先指定需要参与优化的参数块对象
    ceres::Manifold* quaternion_parameterization =
        new QuaternionParameterization;
    problem.AddParameterBlock(q, 4, quaternion_parameterization);

    for (size_t i = 0; i < v_bis.size(); ++i) {
#if USE_AUTO_DIFF
        ceres::CostFunction* cost_function =
            new ceres::AutoDiffCostFunction<RotationResidual, 3, 4>(
                new RotationResidual(v_bis[i], v_lis[i]));
#else
        ceres::CostFunction* cost_function =
            new RotationResidual(v_bis[i], v_lis[i]);
#endif
        // 2、再指定残差块，这样就不需要再后续指定problem.SetManifold(q,
        // quaternion_parameterization)
        problem.AddResidualBlock(cost_function, nullptr, q);
    }

    // 配置优化选项
    ceres::Solver::Options options;
    options.minimizer_progress_to_stdout = true;
    options.max_num_iterations = 100;
    options.linear_solver_type = ceres::DENSE_QR;

#if USE_AUTO_DIFF
    options.minimizer_type = ceres::TRUST_REGION;
    // options.minimizer_type = ceres::LINE_SEARCH;
#else
    // options.minimizer_type = ceres::LINE_SEARCH;
    options.minimizer_type = ceres::TRUST_REGION;
    options.trust_region_strategy_type = ceres::LEVENBERG_MARQUARDT;
    options.logging_type =
        ceres::PER_MINIMIZER_ITERATION;  // 设置输出log便于bug排查

    // Openvins 配置
    // options.linear_solver_type = ceres::DENSE_SCHUR;
    // options.trust_region_strategy_type = ceres::LEVENBERG_MARQUARDT;
    // // options.linear_solver_type = ceres::SPARSE_SCHUR;
    // // options.trust_region_strategy_type = ceres::LEVENBERG_MARQUARDT;
    // // options.preconditioner_type = ceres::SCHUR_JACOBI;
    // // options.linear_solver_type = ceres::ITERATIVE_SCHUR;
    // // options.minimizer_progress_to_stdout = true;
    // // options.linear_solver_ordering = ordering;
    // options.function_tolerance = 1e-5;
    // options.gradient_tolerance = 1e-4 * options.function_tolerance;
    // 禁止调用glog??
    // options.logging_type = ceres::LoggingType::SILENT;

#endif

    // 运行优化
    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);

    // 输出结果
    std::cout << summary.BriefReport() << std::endl;
    std::cout << "Optimized quaternion (w, x, y, z): " << q[0] << ", " << q[1]
              << ", " << q[2] << ", " << q[3] << std::endl;

    // 将四元数转换为旋转矩阵
    Eigen::Quaterniond q_rot(q[0], q[1], q[2], q[3]);
    Eigen::Matrix3d R = q_rot.toRotationMatrix();
    std::cout << "Optimized rotation matrix:\n" << R << std::endl;

    std::cout << "R*Rbv.T:\n" << R * Rbv.transpose() << endl;

    return 0;
}
