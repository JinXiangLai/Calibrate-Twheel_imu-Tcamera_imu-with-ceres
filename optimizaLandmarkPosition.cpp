
#include "common.h"

using namespace std;

#define USE_AUTO_DIFF 1

#if USE_AUTO_DIFF
// 定义误差函数，自动求导条件下，不需定义Evaluate函数，但必须提供operator()函数以计算残差
struct ProjectionResidual {
    ProjectionResidual(const Eigen::Matrix3d& _Rc_w,
                       const Eigen::Vector3d& _Pc_w,
                       const Eigen::Vector2d& _obv, const Eigen::Matrix3d& _K);

    // 使用自动求导时才必须提供该函数
    template <typename T>
    bool operator()(const T* const _Pw, T* residual) const;

   private:
    Eigen::Matrix3d Rc_w_;
    Eigen::Vector3d Pc_w_;
    Eigen::Vector2d obv_;
    Eigen::Matrix3d K;
};
#else

class ProjectionResidual : public ceres::CostFunction {
   public:
    ProjectionResidual(const Eigen::Matrix3d& _Rc_w,
                       const Eigen::Vector3d& _Pc_w,
                       const Eigen::Vector2d& _obv, const Eigen::Matrix3d& _K);
    virtual bool Evaluate(double const* const* parameters, double* residuals,
                          double** jacobians) const override;

   private:
    Eigen::Matrix3d Rc_w_;
    Eigen::Vector3d Pc_w_;
    Eigen::Vector2d obv_;
    Eigen::Matrix3d K;
};
#endif

Eigen::Matrix3d RPY2Rotation(const Eigen::Vector3d& _rpy,
                             const bool isDeg = true);

Eigen::Vector2d ProjectPw2Pixel(const Eigen::Vector3d& Pw,
                                const Eigen::Matrix3d& Rcw,
                                const Eigen::Vector3d& Pcw);

Eigen::Vector3d EstimatePwInitialValue(const vector<Eigen::Matrix3d>& Rcw,
                                       const vector<Eigen::Vector3d>& Pcw,
                                       const vector<Eigen::Vector2d>& obv,
                                       const ::Eigen::Matrix3d& K);

class SolveLandmarkPosition {
   public:
    SolveLandmarkPosition(const vector<Eigen::Matrix3d>& Rc_w,
                          const vector<Eigen::Vector3d>& Pc_w,
                          const vector<Eigen::Vector2d>& obv,
                          const Eigen::Matrix3d& _K,
                          const Eigen::Vector3d& priorPw = {0., 0., 0.});
    SolveLandmarkPosition() = delete;
    Eigen::Vector3d Optimize();
    Eigen::Matrix3d EstimateCovariance();
    Eigen::Vector3d GetPriorPw() const { return priorPw_; }

   private:
    vector<Eigen::Matrix3d> Rc_w_;
    vector<Eigen::Vector3d> Pc_w_;
    vector<Eigen::Vector2d> obv_;
    Eigen::Vector3d priorPw_;  // 待估计变量
    Eigen::Vector3d optPw_;
    ceres::Problem problem_;
    Eigen::Matrix3d K_;
};

int main() {
    // 仿真真实地图点位置
    const Eigen::Vector3d Pw(20, 30, 40);
    // 仿真观测到上述地图点的相机位姿
    const vector<Eigen::Vector3d> vecAngleRwc{{1, 2, 5},
                                              {
                                                  0,
                                                  1,
                                                  10,
                                              },
                                              {2, 3, 15}};
    const vector<Eigen::Vector3d> vecPw{{2, 3, 4}, {3, 4, 7}, {4, 3, 8}};

    vector<Eigen::Matrix3d> Rc_w;
    for (int i = 0; i < vecAngleRwc.size(); ++i) {
        Rc_w.push_back(RPY2Rotation(vecAngleRwc[i]).transpose());
    }
    vector<Eigen::Vector3d> Pc_w;
    for (int i = 0; i < vecPw.size(); ++i) {
        Pc_w.push_back(-Rc_w[i] * vecPw[i]);
    }

    constexpr int loopTime = 10;  // 保留历史位姿，并使用EKF进行位置更新
    vector<Eigen::Vector3d> historyOptPw;
    vector<Eigen::Vector3d> historyInitPw;
    vector<Eigen::Matrix3d> historyCov;
    for (int t = 0; t < loopTime; ++t) {

        vector<Eigen::Vector2d> obv;
        random_device rd;
        mt19937 gen(rd());
        // 像素噪声标准差为5个像素
        const double std = 5.0;
        normal_distribution<double> dist(0.0, std);
        cv::Mat img(kImgHeight, kImgWidth, CV_8UC3, {0, 0, 0});
        // 产生一些图像观测，并添加噪声
        for (int i = 0; i < vecPw.size(); ++i) {
            // 这里其实也要考虑位姿的扰动误差
            const Eigen::Vector2d _obv = ProjectPw2Pixel(Pw, Rc_w[i], Pc_w[i]);

            const Eigen::Vector2d noise{dist(gen), dist(gen)};
            const Eigen::Vector2d _obv_noisy = _obv + noise;
            obv.push_back(_obv_noisy);
            // obv.push_back(_obv);
            cout << "noise: " << noise.transpose() << endl;
            cout << "obv: " << _obv.transpose() << endl;
            cv::circle(img, cv::Point(_obv[0], _obv[1]), 1,
                       cv::Scalar(0, 255, 0), -1);
            cv::circle(img, cv::Point_(_obv_noisy[0], _obv_noisy[1]), 1,
                       {0, 0, 255}, -1);
            cv::imshow("img", img);
            cv::waitKey(0);
        }

        SolveLandmarkPosition slp(Rc_w, Pc_w, obv, K);
        const Eigen::Vector3d optPw = slp.Optimize();
        const Eigen::Matrix3d cov = slp.EstimateCovariance();
        const Eigen::Vector3d initPw = slp.GetPriorPw();
        historyInitPw.emplace_back(initPw);

        cout << "initPw: " << initPw.transpose() << endl;
        cout << "optPw: " << optPw.transpose() << endl;

        if (historyOptPw.empty()) {
            historyOptPw.emplace_back(optPw);
            historyCov.emplace_back(cov);
        } else {
            // 使用EKF进行更新
            // 雅可比矩阵，这里只有量测更新，因此设为单位矩阵
            const Eigen::Matrix3d H = Eigen::Matrix3d::Identity();
            const Eigen::Matrix3d Ht = H.transpose();
            const Eigen::Matrix3d& Pp = historyCov.back();
            // 检查矩阵正定性
            Eigen::Matrix3d innovationCov = H * Pp * Ht + cov;
            Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> eigensolver(
                innovationCov);
            if (eigensolver.eigenvalues().minCoeff() <= 0) {
                std::cerr
                    << "Matrix H * Pp * Ht + cov is not positive definite!"
                    << std::endl;
                innovationCov +=
                    1e-6 * Eigen::Matrix3d::Identity();  // 添加抖动
            }

            // 计算卡尔曼增益
            Eigen::Matrix3d Kgain = Pp * Ht * innovationCov.inverse();
            if (Kgain.array().abs().maxCoeff() > 1e3) {
                std::cerr << "Kgain is too large, limiting its value."
                          << std::endl;
                Kgain = Kgain.cwiseMin(1e3).cwiseMax(-1e3);
            }

            // 更新状态
            Eigen::Vector3d deltaPw = optPw - historyOptPw.back();
            if (deltaPw.norm() > 1e3) {
                std::cerr << "Delta Pw is too large, limiting its value."
                          << std::endl;
                deltaPw = deltaPw.normalized() * 1e3;
            }
            const Eigen::Vector3d upPw = historyOptPw.back() + Kgain * deltaPw;

            Eigen::Matrix3d Pu = (Eigen::Matrix3d::Identity() - Kgain * H) * Pp;
            historyOptPw.emplace_back(upPw);
            cout << "upPw: " << upPw.transpose() << endl;
            Pu = 0.5 * Pu + 0.5 * Pu.transpose().eval();
            historyCov.emplace_back(Pu);
        }

        cout << endl << endl;
    }

    // 计算重投影残差，无意义，因为如果输入就是错的，你去匹配输入是无用的，或者有先验
    // 即地面的值应该接近于0

    // 输出结果
    const Eigen::Matrix3d& cov = historyCov.back();
    cout << "truePw: " << Pw.transpose() << endl;
    cout << "initPw: " << historyInitPw.back().transpose() << endl;
    cout << "optPw: " << historyOptPw.back().transpose() << endl;
    printf("1sigma: x, y, z: %f, %f, %f\n", sqrt(cov.row(0)[0]),
           sqrt(cov.row(1)[1]), sqrt(cov.row(2)[2]));
    return 0;
}

Eigen::Matrix3d RPY2Rotation(const Eigen::Vector3d& _rpy, const bool isDeg) {
    Eigen::Vector3d rpy;
    if (isDeg)
        rpy = _rpy * kDeg2Rad;
    const Eigen::Matrix3d R =
        Eigen::AngleAxisd(rpy.z(), Eigen::Vector3d::UnitZ()) *
        Eigen::AngleAxisd(rpy.y(), Eigen::Vector3d::UnitY()) *
        Eigen::AngleAxisd(rpy.x(), Eigen::Vector3d::UnitX()).toRotationMatrix();
    return R;
}

Eigen::Vector2d ProjectPw2Pixel(const Eigen::Vector3d& Pw,
                                const Eigen::Matrix3d& Rcw,
                                const Eigen::Vector3d& Pcw) {
    const Eigen::Vector3d Pc = Rcw * Pw + Pcw;
    const Eigen::Vector3d Pn = Pc / Pc[2];
    const Eigen::Vector2d obv = K.block(0, 0, 2, 3) * Pn;
    return obv;
}

Eigen::Vector3d EstimatePwInitialValue(const vector<Eigen::Matrix3d>& Rcw,
                                       const vector<Eigen::Vector3d>& Pcw,
                                       const vector<Eigen::Vector2d>& obv,
                                       const ::Eigen::Matrix3d& K) {
    // px2 = ρ2 * K * [Rcw, Pcw] * Pw
    // [px2]x * px2 = [px2]x * K * [Rcw, Pcw] * Pw = 0
    Eigen::MatrixXd A;
    A.resize(3 * Rcw.size(), 4);
    for (int i = 0; i < Rcw.size(); ++i) {
        const Eigen::Vector3d obv_i{obv[i](0), obv[i](1), 1.0};
        Eigen::Matrix<double, 3, 4> T;
        T.block(0, 0, 3, 3) = Rcw[i];
        T.block(0, 3, 3, 1) = Pcw[i];
        A.block(i * 3, 0, 3, 4) = skew(obv_i) * K * T;
    }
    Eigen::JacobiSVD svd(A, Eigen::ComputeFullV);
    const Eigen::Vector4d bestV =
        svd.matrixV().col(svd.singularValues().size() - 1);
    const Eigen::Vector3d estPw = bestV.head(3) / bestV[3];
    // cout << "A:\n" << A << endl;
    // cout << "singularValues: " << svd.singularValues().transpose() << endl;
    // cout << "est Pw: " << estPw.transpose() << endl;
    return estPw;
}

SolveLandmarkPosition::SolveLandmarkPosition(
    const vector<Eigen::Matrix3d>& Rc_w, const vector<Eigen::Vector3d>& Pc_w,
    const vector<Eigen::Vector2d>& obv, const Eigen::Matrix3d& _K,
    const Eigen::Vector3d& priorPw)
    : Rc_w_(Rc_w), Pc_w_(Pc_w), obv_(obv), priorPw_(priorPw), K_(_K) {}

Eigen::Vector3d SolveLandmarkPosition::Optimize() {
    if (priorPw_.isApprox(Eigen::Vector3d::Zero())) {
        optPw_ = EstimatePwInitialValue(Rc_w_, Pc_w_, obv_, K_);
        priorPw_ = optPw_;
    } else {
        optPw_ = priorPw_;
    }

    problem_.AddParameterBlock(optPw_.data(), 3);

    for (size_t i = 0; i < obv_.size(); ++i) {
#if USE_AUTO_DIFF
        ceres::CostFunction* cost_function =
            new ceres::AutoDiffCostFunction<ProjectionResidual, 2, 3>(
                new ProjectionResidual(Rc_w_[i], Pc_w_[i], obv_[i], K_));
#else
        ceres::CostFunction* cost_function =
            new ProjectionResidual(Rc_w_[i], Pc_w_[i], obv_[i], K_);
#endif
        // 2、再指定残差块，这样就不需要再后续指定problem.SetManifold(q,
        // quaternion_parameterization)
        // 距离越远，容忍的像素误差越小
        ceres::LossFunction* huber = new ceres::HuberLoss(0.001);
        problem_.AddResidualBlock(cost_function, huber, optPw_.data());
    }

    // 配置优化选项
    ceres::Solver::Options options;
    options.minimizer_progress_to_stdout = true;
    options.max_num_iterations = 100;
    options.linear_solver_type = ceres::DENSE_QR;
    options.minimizer_type = ceres::TRUST_REGION;
    options.trust_region_strategy_type = ceres::LEVENBERG_MARQUARDT;
    options.logging_type =
        ceres::PER_MINIMIZER_ITERATION;  // 设置输出log便于bug排查
    options.function_tolerance = 1e-12;
    options.inner_iteration_tolerance = 1e-6;

    // 运行优化
    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem_, &summary);
    return optPw_;
}

Eigen::Matrix3d SolveLandmarkPosition::EstimateCovariance() {
    ceres::Covariance::Options options_cov;
    options_cov.null_space_rank = 0;  // 不为0时在信息矩阵不可逆时才能计算伪逆
    // 最小互反条件数
    options_cov.min_reciprocal_condition_number = 1e-10;
    // 使用SVD分解计算伪逆，虽不要求满秩，但是非常耗时 1.0s+
    options_cov.algorithm_type = ceres::CovarianceAlgorithmType::DENSE_SVD;
    // options_cov.algorithm_type = ceres::CovarianceAlgorithmType::SPARSE_QR;
    options_cov.apply_loss_function =
        true;  // Better consistency if we use this
    options_cov.num_threads = 1;
    // Compute covariance for the parameter block p
    ceres::Covariance covariance(options_cov);

    Eigen::Matrix<double, 3, 3, Eigen::RowMajor> covMatrix;
    covMatrix.setZero();
    if (!covariance.Compute({optPw_.data()}, &problem_)) {
        cerr << "Failed to compute covariance." << endl;
    } else {
        covariance.GetCovarianceBlock(optPw_.data(), optPw_.data(),
                                      covMatrix.data());

        // Output the covariance matrix
        cout << "Covariance matrix for p:\n" << covMatrix << endl;
    }
    return covMatrix;
}

#if USE_AUTO_DIFF
ProjectionResidual::ProjectionResidual(const Eigen::Matrix3d& _Rc_w,
                                       const Eigen::Vector3d& _Pc_w,
                                       const Eigen::Vector2d& _obv,
                                       const Eigen::Matrix3d& _K)
    : Rc_w_(_Rc_w), Pc_w_(_Pc_w), obv_(_obv), K(_K) {}

template <typename T>
bool ProjectionResidual::operator()(const T* const _Pw, T* residual) const {
    const Eigen::Matrix<T, 3, 3> Rcw = Rc_w_.cast<T>();
    const Eigen::Matrix<T, 3, 1> Pcw = Pc_w_.cast<T>();
    const Eigen::Matrix<T, 3, 1> Pw(_Pw[0], _Pw[1], _Pw[2]);

    // 重投影
    const Eigen::Matrix<T, 3, 1> Pc = Rcw * Pw + Pcw;
    const Eigen::Matrix<T, 3, 1> Pn = Pc / Pc[2];
    const Eigen::Matrix<T, 2, 1> obv = K.block(0, 0, 2, 3) * Pn;

    // 计算残差
    residual[0] = obv(0) - obv_(0);
    residual[1] = obv(1) - obv_(1);
    return true;
}
#else
ProjectionResidual::ProjectionResidual(const Eigen::Matrix3d& _Rc_w,
                                       const Eigen::Vector3d& _Pc_w,
                                       const Eigen::Vector2d& _obv,
                                       const Eigen::Matrix3d& _K)
    : Rc_w_(_Rc_w), Pc_w_(_Pc_w), obv_(_obv), K(_K) {
    // SizedCostFunction同样继承自CostFunction，
    // 若不使用SizedCostFunction，则需要手动设置残差维度和参数块大小
    set_num_residuals(2);
    // 这里
    mutable_parameter_block_sizes()->push_back(3);
}

bool ProjectionResidual::Evaluate(double const* const* parameters,
                                  double* residuals, double** jacobians) const {
    // 获取优化参数
    const double* p = parameters[0];
    Eigen::Vector3d Pw(p[0], p[1], p[2]);

    // 重投影
    const Eigen::Matrix<double, 3, 1> Pc = Rc_w_ * Pw + Pc_w_;
    const Eigen::Matrix<double, 3, 1> Pn = Pc / Pc[2];
    const Eigen::Matrix<double, 2, 1> obv = K.block(0, 0, 2, 3) * Pn;

    // 计算残差
    residuals[0] = obv(0) - obv_(0);
    residuals[1] = obv(1) - obv_(1);

    // 计算雅可比
    if (jacobians) {
        if (jacobians[0]) {
            // Evaluate函数中的雅可比是关于环境维度的，所以维度是[3x4]
            Eigen::Map<Eigen::Matrix<double, 2, 3, Eigen::RowMajor>> j(
                jacobians[0]);
            j.setZero();
            const Eigen::Matrix<double, 2, 3> J_r_Pn = K.block(0, 0, 2, 3);
            const double invZ = 1 / Pc[2];
            const double invZ2 = invZ * invZ;
            const Eigen::Matrix3d J_Pn_Pc =
                (Eigen::Matrix3d() << invZ, 0, -Pc[0] * invZ2, 0, invZ,
                 -Pc[1] * invZ2, 0, 0, 0)
                    .finished();
            j = J_r_Pn * J_Pn_Pc * Rc_w_;
        }
    }

    return true;
}
#endif