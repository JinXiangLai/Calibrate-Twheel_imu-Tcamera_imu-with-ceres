#include "ceres_problem.h"


using namespace std;

#define USE_AUTO_DIFF 0
#define ROT_RES_DIM 3

// 将旋转矩阵转换为旋转向量，BCH近似时使用
Eigen::Vector3d LogSO3(const Eigen::Matrix3d &R) {
    // 根据3维旋转矩阵R与轴角表示的换算关系，迹迹tr(R) = 1 + 2cos(theta)
    // 因此可以通过迹来计算旋转角
    const double tr = R(0, 0) + R(1, 1) + R(2, 2);
    // 获取旋转轴可以将旋转矩阵转为四元数再提取，
    // 亦可已根据：R对应的旋转轴在经过R旋转后不变的性质，即：
    // 旋转矩阵R的一个特征值始终为1，对应的特征向量即为旋转轴，
    // R*w=w来求解。
    
    // 这里的w实际不是旋转轴a，而是： w=sin(θ)*a
    // 因此最终返回值需要除以sin(θ)，并且：
    // 旋转轴一定是归一化的旋转向量
    Eigen::Vector3d w;
    w << (R(2, 1) - R(1, 2)) / 2, (R(0, 2) - R(2, 0)) / 2,
        (R(1, 0) - R(0, 1)) / 2;
    
    const double costheta = (tr - 1.0) * 0.5f;
    if (costheta > 1 || costheta < -1) return w;
    const double theta = acos(costheta);
    const double s = sin(theta);
    // θ角度值很小时,sin(θ)≈θ, 因此：
    // 直接返回轴角表示即：w=sin(θ)*a≈θ*a
    // 这里的a是归一化的旋转向量
    if (fabs(s) < 1e-5)
        return w;
    else
        // θ角度值较大时
        // 返回θ*(a*sin(θ)/sin(θ))=θ*a
        return theta * w / s; // θ/sin(θ)*w
}

// BCH近似时使用
Eigen::Matrix3d InverseRightJacobianSO3(const double x, const double y,
                                        const double z) {
    // 根据轴角表示法含义即:w=θ*a,a表示归一化向量，
    // 这里的d表示的就是角度θ
    const double d2 = x * x + y * y + z * z;
    const double d = sqrt(d2);

    Eigen::Matrix3d W;
    W << 0.0, -z, y, z, 0.0, -x, -y, x, 0.0;
    if (d < 1e-5)
        // 此时有sin(θ)≈θ, 因此：
        // 1.0 / d2 - (1.0 + cos(d)) / (2.0 * d * sin(d))≈0
        return Eigen::Matrix3d::Identity();
    else
        return Eigen::Matrix3d::Identity() + W / 2 +
               W * W * (1.0 / d2 - (1.0 + cos(d)) / (2.0 * d * sin(d)));
}

Eigen::Matrix3d InverseRightJacobianSO3(const Eigen::Vector3d &v) {
    return InverseRightJacobianSO3(v[0], v[1], v[2]);
}

class Pose {
   public:
    Pose(const Eigen::Quaterniond &q_wb, const Eigen::Vector3d &p_wb)
        : q_wb_(q_wb), p_wb_(p_wb) {}
    Pose(const Eigen::Vector3d &rpy, const Eigen::Vector3d &pos,
         const bool isAngle = true)
        : p_wb_(pos) {
        Eigen::Vector3d r = rpy;
        if (isAngle) {
            r = rpy * M_PI / 180;
        }
        q_wb_ = Eigen::AngleAxisd(r[2], Eigen::Vector3d::UnitZ()) *
                Eigen::AngleAxisd(r[1], Eigen::Vector3d::UnitY()) *
                Eigen::AngleAxisd(r[0], Eigen::Vector3d::UnitX());
    }
    Pose operator*(const Pose &T) {
        Eigen::Quaterniond q = q_wb_ * T.q_wb_;
        Eigen::Vector3d p = q_wb_ * T.p_wb_ + p_wb_;
        return Pose(q, p);
    }
    Pose Inverse() { return Pose(q_wb_.inverse(), -(q_wb_.inverse() * p_wb_)); }
    Eigen::Quaterniond q_wb_;
    Eigen::Vector3d p_wb_;
};

#if USE_AUTO_DIFF
// 定义误差函数，自动求导条件下，不需定义Evaluate函数，但必须提供operator()函数以计算残差
struct RelativePoseResidual {
    RelativePoseResidual(const Pose &Tb1b2, const Pose &Tc1c2)
        : Tb1b2_(Tb1b2), Tc1c2_(Tc1c2) {}

    // 使用自动求导时才必须提供该函数
    template <typename T>
    bool operator()(const T *const qcb, const T *const pcb, T *residual) const {
        // 将四元数转换为 Eigen::Quaternion
        const Eigen::Quaterniond &q_b1b2 = Tb1b2_.q_wb_;
        const Eigen::Quaterniond &q_c1c2 = Tc1c2_.q_wb_;
        const Eigen::Vector3d &p_b1b2 = Tb1b2_.p_wb_;
        const Eigen::Vector3d &p_c1c2 = Tc1c2_.p_wb_;

        Eigen::Quaternion<T> q_cb(qcb[0], qcb[1], qcb[2], qcb[3]);
        Eigen::Matrix<T, 3, 1> p_cb(pcb[0], pcb[1], pcb[2]);

        // 计算残差 Tcb*Tb1b2 = Tc1c2*Tcb ==>
        // 旋转残差，因为ep也耦合了旋转误差，我们是否可以先使用最小二乘计算一个校准的Rcb，然后再在ep中联合优化呢？
        // 但是最小二乘的本质也是右边的四元数是0向量了
        // er = q_cb*(q_b1b2[r]-q_c1c2[l])
#if ROT_RES_DIM == 4
        const Eigen::Quaternion<T> q1 =
            q_cb * q_b1b2.cast<T>();  // 没有重载减法
        const Eigen::Quaternion<T> q2 = q_c1c2.cast<T>() * q_cb;
        Eigen::Matrix<T, 4, 1> er;
        er << q1.w() - q2.w(), q1.x() - q2.x(), q1.y() - q2.y(),
            q1.z() - q2.z();
#elif ROT_RES_DIM == 3
        Eigen::Quaternion<T> eq = q_cb * Tb1b2_.q_wb_.cast<T>() *
                                  q_cb.inverse() *
                                  Tc1c2_.q_wb_.cast<T>().inverse();
        // const Eigen::Matrix<T, 3, 1> er=LogSO3(eq.template
        // cast<double>().toRotationMatrix()).template cast<T>();
        Eigen::Quaterniond eqd = Eigen::Quaterniond::Identity();
        // eqd.w() = ceres::Jet<double, 7>(q_cb.w()).a;
        // eqd.x() = ceres::Jet<double, 7>(q_cb.x()).a;
        // eqd.y() = ceres::Jet<double, 7>(q_cb.y()).a;
        // eqd.z() = ceres::Jet<double, 7>(q_cb.z()).a;
        eqd.w() = ceres::Jet<double, 7>(eq.w()).a;
        eqd.x() = ceres::Jet<double, 7>(eq.x()).a;
        eqd.y() = ceres::Jet<double, 7>(eq.y()).a;
        eqd.z() = ceres::Jet<double, 7>(eq.z()).a;
        // cout << "eqd: " << eqd.w() << " " << eqd.x() << " " << eqd.y() << " "
        // << eqd.z() << endl;
        Eigen::Vector3d erd = LogSO3(eqd.toRotationMatrix());
        const Eigen::Matrix<T, 3, 1> er = erd.cast<T>();
#endif

        // 平移残差
        const Eigen::Matrix<T, 3, 1> ep =
            q_cb.toRotationMatrix() * p_b1b2.cast<T>() +
            (Eigen::Matrix<T, 3, 3>::Identity() -
             q_c1c2.cast<T>().toRotationMatrix()) *
                p_cb -
            p_c1c2.cast<T>();

        // 计算残差
        // 不能直接赋值，因为residual是double类型的指针，而不是Eigen::Matrix类型的指针
        // Eigen::Map<Eigen::Matrix<T, 7, 1, Eigen::RowMajor>> r(residual);
        // r << er[0], er[1], er[2], er[3], ep[0], ep[1], ep[2];
        for (int i = 0; i < int(ROT_RES_DIM); ++i) {
            residual[i] = T(100) * er[i];
        }
        for (int i = 0; i < 3; ++i) {
            residual[i + int(ROT_RES_DIM)] = ep[i];
        }

        return true;
    }

   private:
    const Pose Tb1b2_;  // IMU轨迹
    const Pose Tc1c2_;  // 相机轨迹
};

#else

// 第2个参数对应Manifold的环境维度还是切空间维度呢？
class RelativePoseResidual : public ceres::CostFunction {
   public:
    RelativePoseResidual(const Pose &Tb1b2, const Pose &Tc1c2)
        : Tb1b2_(Tb1b2), Tc1c2_(Tc1c2) {
        // SizedCostFunction同样继承自CostFunction，
        // 若不使用SizedCostFunction，则需要手动设置残差维度和参数块大小
        set_num_residuals(6);                           // [er, ep]
        mutable_parameter_block_sizes()->push_back(4);  // q_cb
        mutable_parameter_block_sizes()->push_back(3);  // p_cb
    }

    // 使用解析雅可比需要提供Evaluate函数，此时不需要提供operator()函数
    // 所以这里是自动求导和解析雅可比的区别
    // 注意：**parameters这里是二维指针
    virtual bool Evaluate(double const *const *parameters, double *residuals,
                          double **jacobians) const override {
        // Ensure parameters are not null
        if (!parameters[0] || !parameters[1]) {
            std::cerr << "Error: Null parameter pointer." << std::endl;
            return false;
        }

        // Ensure residuals are not null
        if (!residuals) {
            std::cerr << "Error: Null residuals pointer." << std::endl;
            return false;
        }
        // 将四元数转换为 Eigen::Quaternion
        const double *paraQ = parameters[0];
        const double *paramP = parameters[1];
        Eigen::Quaternion<double> q(paraQ[0], paraQ[1], paraQ[2], paraQ[3]);
        Eigen::Vector3d p(paramP[0], paramP[1], paramP[2]);
        // cout << "ceres q: " << q.w() << " " << q.x() << " " << q.y() << " "
        // << q.z() << endl; cout << "ceres p: " << p.transpose() << endl;
        const Eigen::Quaterniond eq =
            q * Tb1b2_.q_wb_ * q.inverse() * Tc1c2_.q_wb_.inverse();
        Eigen::Vector3d er = LogSO3(eq.toRotationMatrix());

        const Eigen::Vector3d ep =
            q * Tb1b2_.p_wb_ +
            (Eigen::Matrix3d::Identity() - Tc1c2_.q_wb_.toRotationMatrix()) *
                p -
            Tc1c2_.p_wb_;

        // 计算残差，修改权重之后，问题可以解决，所以问题的症结在哪里
        // 问题出在推导雅可比，提取公因式时，搞错了一个正负号
        residuals[0] = 100 * er[0];
        residuals[1] = 100 * er[1];
        residuals[2] = 100 * er[2];  // 似乎是该维度的雅可比不准？？
        residuals[3] = 1 * ep[0];
        residuals[4] = 1 * ep[1];
        residuals[5] = 1 * ep[2];

        // cout << "residuals: " << residuals[0] << " " << residuals[1] << " "
        // << residuals[2] << " " << residuals[3] << " " << residuals[4] << " "
        // << residuals[5] << endl;

        // 手动提供解析雅可比矩阵
        // 对于Manifold，这里提供的是关于环境维度的雅可比矩阵
        // 所以不能直接使用李代数的结果来赋值，这里提供∂e/∂q，而不是∂e/∂δ
        // 由于四元数可以直接转为旋转矩阵，因此e=q*v=R_q*v，这样就可以得到e关于[qw,
        // qx, qy, qz]的雅可比矩阵了 但这种做法很明显比较复杂，不是我的首选

        // 因为有些时候如在TRUST_REGION策略下，只需要计算residuals
        if (jacobians) {
            if (jacobians[0]) {
                // Evaluate函数中的雅可比是关于环境维度的，所以维度是[3x4]
                Eigen::Map<Eigen::Matrix<double, 6, 4, Eigen::RowMajor>> j(
                    jacobians[0]);
                // 右乘扰动更新雅可比
                // 这里给出了∂e/∂δ的李代数雅可比，因此要注意PlusJacobian中要求实现∂q/∂δ不再是真实值，
                // 而是要保证该处雅可比j右乘上它，矩阵值可以保持不变。
                // 此外需要注意，不能使用ceres::TRUST_REGION，只能使用ceres::LINE_SEARCH，
                // 因为PlusJacobian中的雅可比矩阵不是真实的，ceres2不能用其计算信任域
                j.setZero();
                j.block(0, 0, 3, 3) =
                    InverseRightJacobianSO3(er) *
                    (Tc1c2_.q_wb_ * q).toRotationMatrix() *
                    (Tb1b2_.q_wb_.inverse().toRotationMatrix() -
                     Eigen::Matrix3d::Identity());

                // 只给出这个雅可比时，优化结果也近似正确，所以问题是上述关于旋转的雅可比存在问题？
                j.block(3, 0, 3, 3) =
                    -q.toRotationMatrix() *
                    skew(
                        Tb1b2_
                            .p_wb_);  // 关闭以只优化旋转，测试所推导旋转雅可比的正确性
            }

            if (jacobians[1]) {
                Eigen::Map<Eigen::Matrix<double, 6, 3, Eigen::RowMajor>> j(
                    jacobians[1]);
                j.setZero();
                // j.block(3, 3, 3, 3) =
                // (Eigen::Matrix3d::Identity()-Tc1c2_.q_wb_.toRotationMatrix());
                // 注意，这里jacobians[1]仅是对于第1个状态量的雅可比，最多仅有3列，而不是6列，所以上述写法出错！！！
                j.block(3, 0, 3, 3) =
                    (Eigen::Matrix3d::Identity() -
                     Tc1c2_.q_wb_
                         .toRotationMatrix());  // 关闭以只优化旋转，测试所推导旋转雅可比的正确性
            }
        }

        return true;
    }

   private:
    const Pose Tb1b2_;  // IMU轨迹
    const Pose Tc1c2_;  // 相机轨迹
};
#endif

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
