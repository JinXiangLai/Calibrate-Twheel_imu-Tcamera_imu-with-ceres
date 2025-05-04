#include "ceres_problem.h"

using namespace std;


int main() {
    vector<Eigen::Vector3d> rpy{{10, 20, 30}, {5, 9, 11}, {-60, 30, 0}};
    vector<Eigen::Vector3d> pos{{1, 2, 3}, {0, 4, 5}, {10, 10, 10}};
    Pose Tcb(Eigen::Vector3d(5, 6, 35), Eigen::Vector3d(0.5, 2, 3));
    Pose invTcb = Tcb.Inverse();
    vector<Pose> Tb1b2, Tc1c2;
    for (int i = 0; i < rpy.size(); ++i) {
        Tb1b2.push_back(Pose(rpy[i], pos[i]));
        Tc1c2.push_back(Tcb * Tb1b2[i] * invTcb);
    }

    // 初始四元数（单位四元数）
    Pose estTcb({20, 50, 300}, {0, 1, 0});
    // estTcb = Tcb;

    // 构建优化问题
    ceres::Problem problem;
    // 1、先指定需要参与优化的参数块对象
    ceres::Manifold *quaternion_parameterization =
        new QuaternionParameterization;

    double *q = estTcb.q_wb_.coeffs()
                    .data();  // 返回的是[qx, qy, qz, qw]
                              // 因此最后的结果会有错误，除非手动重新赋值
    Eigen::Quaterniond qu = estTcb.q_wb_;
    cout << "quat.coeffs().data()返回顺序确认：" << endl;
    cout << "q: " << q[0] << " " << q[1] << " " << q[2] << " " << q[3] << endl;
    cout << "qu: " << qu.w() << " " << qu.x() << " " << qu.y() << " " << qu.z()
         << endl;

    double qq[4] = {qu.w(), qu.x(), qu.y(), qu.z()};
    // q=qq;
    problem.AddParameterBlock(q, 4, quaternion_parameterization);
    problem.AddParameterBlock(estTcb.p_wb_.data(), 3);
    vector<double *>
        parameters;  // 存放double*，意味着你需要记录下每个double*对应的元素个数，就如Openvins一样，否则你只能在后续挨个恢复元素？
    parameters.push_back(q);
    parameters.push_back(estTcb.p_wb_.data());

    cout << "测试放在一起的parameters:" << endl;
    for (int i = 0; i < 7; ++i) {
        if (i < 4)
            cout << parameters[0][i] << " ";  // 容器的第一个指针有4个元素
        else {
            cout << parameters[1][i - 4] << " ";  // 容器的第2个指针有3个元素
        }
    }
    cout << endl;

    for (size_t i = 0; i < Tb1b2.size(); ++i) {
#if USE_AUTO_DIFF
        ceres::CostFunction *cost_function =
            new ceres::AutoDiffCostFunction<RelativePoseResidual, (ROT_RES_DIM + 3),
                                            4, 3>(
                new RelativePoseResidual(Tb1b2[i], Tc1c2[i]));
        problem.AddResidualBlock(cost_function, nullptr, q,
                                 estTcb.p_wb_.data());
#else
        ceres::CostFunction *cost_function =
            new RelativePoseResidual(Tb1b2[i], Tc1c2[i]);
        problem.AddResidualBlock(cost_function, nullptr, parameters);
#endif
        // 2、再指定残差块，这样就不需要再后续指定problem.SetManifold(q,
        // quaternion_parameterization)
    }

    // problem.SetParameterBlockConstant(estTcb.p_wb_.data()); // 不优化平移

    // 配置优化选项
    ceres::Solver::Options options;
    options.minimizer_progress_to_stdout = true;
    options.max_num_iterations = 100;
    options.linear_solver_type = ceres::DENSE_QR;

#if USE_AUTO_DIFF
    options.minimizer_type = ceres::TRUST_REGION;
    // options.minimizer_type = ceres::LINE_SEARCH;
#else
    // options.max_line_search_step_contraction = 1e-3;
    // options.min_line_search_step_contraction = 0.6;

    // // LBFGS/BFGS线性搜索只能使用WOLFE
    // //
    // BFGS可以使H_k矩阵保持正定的证明：https://xbna.pku.edu.cn/fileup/0479-8023/HTML/2020-6-1013.html
    // 不好用，当我使用er=100*er,
    // ep=ep时，线性搜索无法收敛，而TRUST_REGION可以收敛 options.minimizer_type
    // = ceres::LINE_SEARCH; options.line_search_direction_type = ceres::LBFGS;
    // options.line_search_type = ceres::WOLFE;
    // 下面这种共轭配置无法收敛
    // options.line_search_direction_type = ceres::NONLINEAR_CONJUGATE_GRADIENT;
    // options.line_search_type = ceres::ARMIJO;

    // 以下trust region对于直接给定雅可比应该是不可行的
    options.minimizer_type = ceres::TRUST_REGION;
    options.trust_region_strategy_type = ceres::LEVENBERG_MARQUARDT;
    // options.trust_region_strategy_type = ceres::DOGLEG;
#endif

    // 运行优化
    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);

    // 输出结果
    std::cout << summary.BriefReport() << std::endl;
    // std::cout << summary.FullReport() << std::endl;

    std::cout << "Optimized quaternion (w, x, y, z): " << q[0] << ", " << q[1]
              << ", " << q[2] << ", " << q[3] << std::endl;

    // 将四元数转换为旋转矩阵
    Eigen::Quaterniond q_rot(q[0], q[1], q[2], q[3]);
    estTcb.q_wb_ = q_rot;  // 由于coffes().data()返回顺序问题，故而需要重新赋值

    std::cout << "estR*Rcb.T:\n"
              << (estTcb.q_wb_ * Tcb.q_wb_.inverse()).toRotationMatrix()
              << endl;
    std::cout << "estP - Tcb.p_wb_: " << (estTcb.p_wb_ - Tcb.p_wb_).transpose()
              << endl;

    return 0;
}
