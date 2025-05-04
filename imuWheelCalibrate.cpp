#include "ceres_problem.h"

using namespace std;


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
