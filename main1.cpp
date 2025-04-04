 
#include <ceres/ceres.h>
#include <Eigen/Dense>
#include <iostream>
#include <vector>

// 定义误差函数
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

// 自定义四元数的 LocalParameterization
class QuaternionParameterization : public ceres::LocalParameterization {
public:
    virtual bool Plus(const double* x, const double* delta, double* x_plus_delta) const {
        // 将 delta 转换为四元数
        Eigen::Quaterniond delta_q(1.0, delta[0], delta[1], delta[2]);

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
        // 四元数的 Jacobian 矩阵
        Eigen::Map<Eigen::Matrix<double, 4, 3, Eigen::RowMajor>> j(jacobian);
        j.setZero();
        j(1, 0) = 1.0;
        j(2, 1) = 1.0;
        j(3, 2) = 1.0;

        return true;
    }

    virtual int GlobalSize() const { return 4; }  // 四元数的全局维度
    virtual int LocalSize() const { return 3; }   // 四元数的局部维度（李代数维度）
};

int main() {
    // 示例数据：IMU 和轮速计的速度测量值
    std::vector<Eigen::Vector3d> v_bis = {
        {1.0, 0.0, 0.0},
        {0.0, 1.0, 0.0},
        {0.0, 0.0, 1.0}
    };

    std::vector<Eigen::Vector3d> v_lis = {
        {0.9, 0.1, 0.0},
        {0.1, 0.9, 0.0},
        {0.0, 0.0, 1.1}
    };

	Eigen::Matrix3d Rbv = Eigen::AngleAxisd(0.13, Eigen::AngleAxisd::Vector3::UnitY()) *
							Eigen::AngleAxisd(1.3, Eigen::AngleAxisd::Vector3::UnitX()).toRotationMatrix();
	for(int i=0; i<v_bis.size(); ++i) {
		v_bis[i] = Rbv*v_lis[i];
	}

    // 初始四元数（单位四元数）
    double q[4] = {1.0, 0.0, 0.0, 0.0};  // 初始化为无旋转

    // 构建优化问题
    ceres::Problem problem;
    for (size_t i = 0; i < v_bis.size(); ++i) {
        ceres::CostFunction* cost_function =
            new ceres::AutoDiffCostFunction<RotationResidual, 3, 4>(
                new RotationResidual(v_bis[i], v_lis[i]));
        problem.AddResidualBlock(cost_function, nullptr, q);
    }

    // 使用 LocalParameterization 处理四元数的单位约束
    ceres::LocalParameterization* quaternion_parameterization = new QuaternionParameterization;
    problem.SetParameterization(q, quaternion_parameterization);

    // 配置优化选项
    ceres::Solver::Options options;
    options.minimizer_progress_to_stdout = true;
    options.max_num_iterations = 100;
    options.linear_solver_type = ceres::DENSE_QR;

    // 运行优化
    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);

    // 输出结果
    std::cout << summary.BriefReport() << std::endl;
    std::cout << "Optimized quaternion (w, x, y, z): "
              << q[0] << ", " << q[1] << ", " << q[2] << ", " << q[3] << std::endl;

    // 将四元数转换为旋转矩阵
    Eigen::Quaterniond q_rot(q[0], q[1], q[2], q[3]);
    Eigen::Matrix3d R = q_rot.toRotationMatrix();
    std::cout << "Optimized rotation matrix:\n" << R << std::endl;

	std::cout << "R*Rbv.T:\n" << R*Rbv.transpose();

    return 0;
}

    