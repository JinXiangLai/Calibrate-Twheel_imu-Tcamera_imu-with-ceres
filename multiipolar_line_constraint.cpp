/*
    多极线约束仿真验证程序
*/

#include <deque>
#include <random>

#include "ceres_problem.h"
#include "classes.h"
#include "common.hpp"

using namespace std;

constexpr double huberTH = 15.99;
constexpr double noiseStd = 3.0;
constexpr double noiseStd2 = noiseStd * noiseStd;
#if RESIDUAL_ON_NORMLIZED_PLANE
const double kRatio = kSqrt2 / (sqrt(kFx * kFx + kFy * kFy));
const double huberTHnorm = kRatio * huberTH;
const double noiseStdNorm = kRatio * noiseStd;
const double noiseStdNorm2 = noiseStdNorm * noiseStdNorm;
#endif

constexpr double INF = 1e12;
constexpr int ceresMaxIterativeTime = 1000;

// 每次量测更新时所需要的最少帧数量，比这个更重要的是基线长度
constexpr int kMinUpdateFrameNumEachTime = 9;
// 姿态、观测条件都一样的情况下，最影响地图点位置不确定度的只有基线
// 其次就是量测的个数了，如果基线长短不断变换，那么标准差自然来回波动，
// 如果累积基线长度不断增大，那么标准差自然不断减小，
// 问题是：如何把先验转换为累积基线，以降低标准差
constexpr double kMinUpdateBaselineInSW = 6.5;
// 一个可行的办法是增大先验的权重，这样就能每次降低不确定度，且必须增长够快，如线性增长
// 才能使std不断下降，这是因为W增大，海森矩阵H增大，而最终的方差与H矩阵的逆正相关
#define ExpandPriorWeightAccordingToAccumulateBaseline 0

class SolveLandmarkPosition {
   public:
    SolveLandmarkPosition(const std::vector<Eigen::Matrix3d>& Rc_w,
                          const std::vector<Eigen::Vector3d>& Pc_w,
                          const std::vector<std::vector<Eigen::Vector2d>>& obvs,
                          const Eigen::Matrix3d& _K,
                          const Eigen::Vector3d& priorPw = {0., 0., 0.});
    SolveLandmarkPosition() = delete;
    Eigen::Vector3d Optimize();
    bool EstimateCovariance(
        Eigen::Matrix<double, 3, 3, Eigen::RowMajor>& covMatrix);
    Eigen::Vector3d GetPriorPw() const { return priorPw_; }
    void SetPriorPw(const Eigen::Vector3d& p) { priorPw_ = p; }
    void SetPriorWeight(const Eigen::Matrix3d& w) { priorWeight_ = w; }

   private:
    std::vector<Eigen::Matrix3d> Rc_w_;
    std::vector<Eigen::Vector3d> Pc_w_;
    std::vector<std::vector<Eigen::Vector2d>> obvs_;
    Eigen::Vector3d priorPw_;  // 待估计变量
    Eigen::Matrix3d priorWeight_ = Eigen::Matrix3d::Zero();
    Eigen::Vector3d optPw_;
    ceres::Problem problem_;
    Eigen::Matrix3d K_;
};

bool SlidingWindowSolvedByCeres(const std::deque<DataFrame>& slidingWindow,
                                const Eigen::Vector3d& priorPw,
                                const PriorEstimateData& priorData,
                                const double& accumulateBaseline,
                                const double& lastUpdateAccBaseline,
                                const int updateTime, Eigen::Vector3d& optPw,
                                Eigen::Matrix3d& cov);

int main(int argc, char** argv) {
    double radius = 3.0;        // 每个marker半径
    double depth = 50.0;        //
    double distX = radius * 4;  // 两个marker距离
    constexpr int kMarkerNum = 2;
    printf("Default parameters:\n\tradius=%f, depth=%f, distX=%f\n", radius,
           depth, distX);
    if (argc > 1) {
        radius = atof(argv[1]);
        distX = radius * 4;
    }
    if (argc > 2) {
        depth = atof(argv[2]);
    }
    printf("Current parameters:\n\tradius=%f, depth=%f, distX=%f\n", radius,
           depth, distX);

    // 初始化两个一模一样的marker点
    // clang-format off
    Eigen::Matrix<double, 3, kMarkerNum> Pws = 
        (Eigen::Matrix<double, 3, kMarkerNum>() << 0, distX,
                                                   0, 0,
                                                   depth, depth).finished();
    // clang-format on

    auto ProjectPws2CurFrame =
        [&Pws, &depth](const Eigen::Matrix3d& Rwc = Eigen::Matrix3d::Identity(),
                       const Eigen::Vector3d& Pwc = {0, 0, 0},
                       const int& time = 0) -> DataFrame {
        const Eigen::Matrix3d Rc_w = Rwc.transpose();
        // c是新原点，因此需要先计算c系原点到w原点的方向c->w，再旋转到c系即可
        const Eigen::Vector3d Pc_w = Rc_w * (Eigen::Vector3d(0, 0, 0) - Pwc);

        // 遍历每一个世界点，计算投影点
        vector<Eigen::Vector2d> obvEachFrame;
        cv::Mat debugImg(kImgHeight, kImgWidth, CV_8UC3);
        debugImg.setTo(cv::Scalar(0, 0, 0));
        for (int j = 0; j < Pws.cols(); ++j) {
            const Eigen::Vector3d& Pw = Pws.col(j);
            // 这里其实也要考虑位姿的扰动误差
            Eigen::Vector2d _obv = ProjectPw2Pixel(Pw, Rc_w, Pc_w, K);
            _obv[0] = int(_obv[0]);
            _obv[1] = int(_obv[1]);

            mt19937 gen(44);
            normal_distribution<double> dist(0.0, noiseStd);
            const Eigen::Vector2d noise{int(dist(gen)), int(dist(gen))};
            const Eigen::Vector2d _obv_noisy = _obv + noise;
            if (j != 0) {
                obvEachFrame.push_back(_obv_noisy);
            } else {
                obvEachFrame.push_back(_obv);
            }

            cv::circle(debugImg, cv::Point2i(_obv[0], _obv[1]), 1,
                       cv::Scalar(0, 255, 0), -1);
            cv::circle(debugImg, cv::Point2i(_obv_noisy[0], _obv_noisy[1]), 1,
                       {0, 0, 255}, -1);
        }
        return DataFrame(Rc_w, Pc_w, depth - Pwc.z(), obvEachFrame, time,
                         debugImg);
    };

    // 初始化世界系的位置
    Eigen::Matrix3d lastRw_c = Eigen::Matrix3d::Identity();
    Eigen::Vector3d lastPw_c = Eigen::Vector3d::Zero();

    // 滑动窗口
    deque<DataFrame> historyFrame = {
        ProjectPws2CurFrame(lastRw_c, lastPw_c, 0)};  // 初始化第一帧
    double accumulateBaseline = 0.0;
    double lastUpdateAccBaseline = 0.0;

    // 设置运动参数
    //const Eigen::Vector2d rotRange(-0.05, 0.1);  // degree
    const Eigen::Vector2d rotRange(2, 5);  // degree

    // TODO: 实验显示，帧间距离越近，不确定度会非常大
    //const Eigen::Vector2d moveRange(0.01, 0.1);  // meter
    // 帧间基线越大，J矩阵值越大(信息越大)，协方差下降越快
    const Eigen::Vector2d moveRange(-0.05, 0.1);  // meter

    int updateTime = 0;
    while (1) {
        ++updateTime;
        cout << endl;
        cout << "last Pwc & Rwc in rpy:\n"
             << "lastPwc: " << lastPw_c.transpose() << "\nlastRwc: "
             << RotationMatrixToZYXEulerAngles(lastRw_c).transpose() * kRad2Deg;
        cout << "Please input " << updateTime << "th Pc1_c2: ";
        Eigen::Vector3d Pc1_c2(0, 0, 0);
        cin >> Pc1_c2[0] >> Pc1_c2[1] >> Pc1_c2[2];
        cout << "Please input " << updateTime << "th Rw_c: ";
        Eigen::Vector3d rpy(0, 0, 0);
        cin >> rpy[0] >> rpy[1] >> rpy[2];
        rpy *= kDeg2Rad;

        // 产生下一个位姿
        Eigen::Matrix3d Rw_c2 = RPY2Rotation(rpy);
        const Eigen::Vector3d Pw_c2 = lastRw_c * Pc1_c2 + lastPw_c;
        accumulateBaseline += Pc1_c2.norm();
        lastRw_c = Rw_c2;
        lastPw_c = Pw_c2;

        cout << "current accumulateBaseline: " << accumulateBaseline << endl;

        DataFrame curFrame = ProjectPws2CurFrame(Rw_c2, Pw_c2, updateTime);

        // 进行极线跟踪，判断特征点1在该图像位置，这个工作量最大

        // 判断是否需要加入该图像到滑窗

        // 不能使用编号，因为sw窗口大小固定
        //cv::imshow("debugImg" + to_string(slidingWindow.size()), debugImg);
        //cv::imshow("debugImg" + to_string(0), debugImg);
        //cv::waitKey(0);

        // 不仅考虑滑窗大小，还得考虑基线
        //if (slidingWindow.size() < kMinUpdateFrameNumEachTime ||
        //    accumulateBaseline - lastUpdateAccBaseline <
        //        kMinUpdateBaselineInSW) {
        if (slidingWindow.size() < kMinUpdateFrameNumEachTime) {
            continue;
        }

        // 显示debug合并可视化信息
        //const cv::Mat mergeImg = stitchAndDrawMatches(slidingWindow);
        //cv::imshow("mergeImg_X0:" + to_string(X0) + "_Y0:" + to_string(Y0),
        //           mergeImg);
        //cv::waitKey(0);

        // 执行一次滑窗优化，并弹出最老一帧
        Eigen::Vector3d priorPw, optPw;
        Eigen::Matrix3d cov = Eigen::Matrix3d::Constant(INF);

        if (!isInitialized) {
            constexpr double maxConditionNumber = 10;
            Eigen::Vector4d singularValues{0, 0, 0, 0};
            CalculateInitialPwDLT(slidingWindow, K, priorPw, singularValues);
            const double conditionNumber =
                singularValues[0] / singularValues[2];
            cout << "condition number: " << conditionNumber << endl;
            vector<Eigen::Matrix3d> Rc_ws;
            vector<Eigen::Vector3d> Pc_ws;
            vector<vector<Eigen::Vector2d>> obvs;
            ExtrackPoseAndObvFromSlidingWindow(slidingWindow, Rc_ws, Pc_ws,
                                               obvs);
            const bool valid = CheckInitialPwValidity(Rc_ws, Pc_ws, priorPw);
            cout << "check init pw valid: " << valid << endl;
            // 奇异值如果远离0, 那么抗噪能力更强
            if ((conditionNumber > maxConditionNumber &&
                 singularValues[2] < 1.0) ||
                !valid) {
                CullingBadObservationsBeforeInit(slidingWindow);
                // TODO: 重置滑窗内相关状态变量
                continue;
            } else {
                // TODO: 输出理论真值的残差
                //PrintReprojectErrorEachFrame(slidingWindow, Pws.col(0), K);
                //const Eigen::Vector3d diffPw =
                //    Pws.col(0) + Eigen::Vector3d(0.5, 0.5, 0.0);
                //PrintReprojectErrorEachFrame(slidingWindow, diffPw, K);
                //PrintReprojectErrorEachFrame(slidingWindow, priorPw, K);
            }
        } else {
            if (!CheckLastestObservationUseful(slidingWindow)) {
                continue;
            }
            priorPw = priorData.GetPw();
        }

        auto t0 = chrono::steady_clock::now();

        const bool success = SlidingWindowSolvedByCeres(
            slidingWindow, priorPw, priorData, accumulateBaseline,
            lastUpdateAccBaseline, updateTime, optPw, cov);

        auto t1 = chrono::steady_clock::now();
        const double spendTime =
            chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count();
        cout << "sliding window num=" << slidingWindow.size()
             << " ceres optimization spend: " << spendTime << "ms!" << endl;

        if (!success) {
            cerr << "solve ceres failed!" << endl;
            continue;
        }

        constexpr double maxMakerPosStd =
            3.0;  // meters，条件数越小，这个相对调大
        // 深度为dm，归一化坐标为(xn, yn, 1)，那么(Xw, Yw) = (d*xn, d*yn)，
        // dm=dt+Δd，那么(Xw, Yw)的误差为(Δd*xn, Δd*yn)，易知，(xn, yn)绝对值不超过，
        // 并且：
        // 1. 像素越靠近中心，(xn, yn)值越小，因此误差Δd引起的(ΔXw, ΔYw)越小；
        // 2. 像素越远离中心，(xn, yn)虽然值越大，但是深度的辨识系数越显著，深度不确定度Δd越小，Δd引起的(ΔXw, ΔYw)越小
        // 我们更关心水平位置精度，因此这里只关心水平位置精度即可
        const Eigen::Vector3d& std = cov.diagonal().array().sqrt().transpose();
        if (std.head(2).norm() > maxMakerPosStd && !isInitialized) {
            cout << "std too big, value: " << std.transpose() << endl;
            CullingBadObservationsBeforeInit(slidingWindow);
            continue;
        }

#define Remove_Unormal_Obv 0
        double chi2 = 0.;
        if (isInitialized) {
            const double square3Sigma = 9;
            chi2 = CalculateChi2Distance(priorData.lastCov_, priorData.lastPw_,
                                         optPw);
            if (chi2 > square3Sigma) {
                cout << "update failed for chi2: " << chi2 << endl;
#if Remove_Unormal_Obv
                slidingWindow.pop_back();
#endif
                continue;
            }

            if (std.squaredNorm() > priorData.lastCov_.diagonal().norm()) {
                // 不更新也不删除观测，因为前面已经检查了观测的有效性
                cout << "update failed for std bigger than last one"
                     << std.transpose() << endl;
#if Remove_Unormal_Obv
                slidingWindow.pop_back();
#endif
                continue;
            }
        }

        isInitialized = true;
        priorData.UpdateMeanAndH(optPw, cov);

        printf("accBaseline - lastAcc: %f-%f=%f\nSW size=%ld\n",
               accumulateBaseline, lastUpdateAccBaseline,
               (accumulateBaseline - lastUpdateAccBaseline),
               slidingWindow.size());
        lastUpdateAccBaseline = accumulateBaseline;
        ++updateTime;
        //PrintReprojectErrorEachFrame(slidingWindow, optPw, K);

        // 移除滑动窗口元素
        auto MoveSlidingWindow = [&slidingWindow]() -> void {
            // 移除与最新帧水平基线最短的帧
            const DataFrame& last = slidingWindow.back();
            int minDistId = -1;
            double minHorizontalBaseline = 1e10;
            for (int i = 0; i < slidingWindow.size() - 1; ++i) {
                const DataFrame& cur = slidingWindow[i];
                // 不一定需要水平位移才能可观，水平位移只是针对位于图像中心的像素而言，对于图像中心像素，
                // 其在归一化平面坐标为(0, 0)，对于方程无贡献
                const double bs = (-cur.Rc_w.transpose() * cur.Pc_w +
                                   last.Rc_w.transpose() * last.Pc_w)
                                      .head(3)
                                      .norm();
                //printf("id=%d, bs=%f\n", i, bs);
                if (bs < minHorizontalBaseline) {
                    minHorizontalBaseline = bs;
                    minDistId = i;
                }
            }
            //printf("midId=%d, minDist=%f\n", minDistId, minHorizontalBaseline);

            slidingWindow.erase(slidingWindow.begin() + minDistId);
            //slidingWindow.pop_front();
        };
        //while (slidingWindow.size() >= kMinUpdateFrameNumEachTime) {
        //    MoveSlidingWindow();
        //}
        // 仅移动一个
        //MoveSlidingWindow();

        // 打印输出
        //cout << "initPw: " << initPw.transpose() << endl;
        cout << "optPw: " << optPw.transpose() << endl;
        cout << "std: " << std.transpose() << endl;
        cout << "chi2: " << to_string(chi2) << endl;
        priorData.lastPw_ = optPw;
        priorData.lastCov_ = cov;
        if (std.norm() < 0.1) {
            cout << "converge at accumulateBaseline=" << accumulateBaseline
                 << endl;
            break;
        }
    }

    return 0;
}

SolveLandmarkPosition::SolveLandmarkPosition(
    const vector<Eigen::Matrix3d>& Rc_w, const vector<Eigen::Vector3d>& Pc_w,
    const vector<vector<Eigen::Vector2d>>& obvs, const Eigen::Matrix3d& _K,
    const Eigen::Vector3d& priorPw)
    : Rc_w_(Rc_w), Pc_w_(Pc_w), obvs_(obvs), priorPw_(priorPw), K_(_K) {}

Eigen::Vector3d SolveLandmarkPosition::Optimize() {
    if (priorPw_.isApprox(Eigen::Vector3d::Zero())) {
        optPw_ = EstimatePwInitialValue(Rc_w_, Pc_w_, obvs_, K_);
        priorPw_ = optPw_;
    } else {
        optPw_ = priorPw_;
    }
    cout << "init optPw: " << optPw_.transpose() << endl;
    problem_.AddParameterBlock(optPw_.data(), 3);

    for (int i = 0; i < Pc_w_.size(); ++i) {
        for (size_t j = 0; j < obvs_[i].size(); ++j) {
            const Eigen::Vector2d& obv = obvs_[i][j];

            ceres::CostFunction* cost_function =
                new ProjectionResidual(Rc_w_[i], Pc_w_[i], obv, K_);

#if RESIDUAL_ON_NORMLIZED_PLANE
            // 2、再指定残差块，这样就不需要再后续指定problem.SetManifold(q,
            // quaternion_parameterization)
            // 距离越远，容忍的像素误差越小
            ceres::LossFunction* huber = new ceres::HuberLoss(huberTHnorm);
#else
            ceres::LossFunction* huber = new ceres::HuberLoss(huberTH);
#endif
            problem_.AddResidualBlock(cost_function, huber, optPw_.data());
        }
    }

    if (!priorWeight_.isApprox(Eigen::Matrix3d::Zero())) {
        ceres::CostFunction* cost_function =
            new PriorPwResidual(priorPw_, priorWeight_);
        problem_.AddResidualBlock(cost_function, nullptr, optPw_.data());
    }

    // 配置优化选项
    ceres::Solver::Options options;
    options.minimizer_progress_to_stdout = true;
    options.max_num_iterations = ceresMaxIterativeTime;
    options.linear_solver_type = ceres::DENSE_QR;
    options.minimizer_type = ceres::TRUST_REGION;
    options.trust_region_strategy_type = ceres::LEVENBERG_MARQUARDT;
    // 设置输出log便于bug排查
    options.logging_type = ceres::SILENT;  // PER_MINIMIZER_ITERATION;
    options.function_tolerance = 1e-12;
    options.inner_iteration_tolerance = 1e-6;

#if RESIDUAL_ON_NORMLIZED_PLANE
    options.max_num_iterations = 50;  // 归一化平面收敛快，可适当减小迭代次数
    options.gradient_tolerance = 1e-12;  // 防止过快终止，适当减小
    options.function_tolerance = 1e-15;  // 适当减小以匹配归一化残差量级
    options.parameter_tolerance = 1e-10;
#endif

    // 运行优化
    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem_, &summary);
    return optPw_;
}

bool SolveLandmarkPosition::EstimateCovariance(
    Eigen::Matrix<double, 3, 3, Eigen::RowMajor>& covMatrix) {
    ceres::Covariance::Options options_cov;
    options_cov.null_space_rank = 0;  // 需要截断的最小奇异值数量
    // 最小互反条件数，即条件数的相反数
    options_cov.min_reciprocal_condition_number =
        kMinReciprocalConditionNumber;  // 1e-6才不会导致方差增大出现，因为已经被过滤掉
    // 使用SVD分解计算伪逆，虽不要求满秩，但是非常耗时 1.0s+
    options_cov.algorithm_type = ceres::CovarianceAlgorithmType::DENSE_SVD;
    // options_cov.algorithm_type = ceres::CovarianceAlgorithmType::SPARSE_QR;
    options_cov.apply_loss_function =
        true;  // Better consistency if we use this
    options_cov.num_threads = 1;
    // Compute covariance for the parameter block p
    ceres::Covariance covariance(options_cov);

    covMatrix.setConstant(INF);
    if (!covariance.Compute({optPw_.data()}, &problem_)) {
        cerr << "Failed to compute covariance！！！" << endl;
        return false;
    } else {
        covariance.GetCovarianceBlock(optPw_.data(), optPw_.data(),
                                      covMatrix.data());
        if (noiseStd2 > 0.0) {
#if RESIDUAL_ON_NORMLIZED_PLANE
            covMatrix *= noiseStdNorm2;
#else
            covMatrix *= noiseStd2;
#endif
        }
        return true;

        // Output the covariance matrix
        //cout << "Covariance matrix for p:\n" << covMatrix << endl;
    }
}

bool SlidingWindowSolvedByCeres(const deque<DataFrame>& slidingWindow,
                                const Eigen::Vector3d& priorPw,
                                const PriorEstimateData& priorData,
                                const double& accumulateBaseline,
                                const double& lastUpdateAccBaseline,
                                const int updateTime, Eigen::Vector3d& optPw,
                                Eigen::Matrix3d& cov) {

    // 利用对地高度计算初值，因为X logo被认为是在地面即d=0，但现在让其为depth=50，
    // 因此我参考系也要往上平移50
    Eigen::Matrix3d priorWeight = priorData.GetH();
    if (!priorWeight.isApprox(Eigen::Matrix3d::Zero())) {

        // 计算信息矩阵的开根号
        Eigen::EigenSolver<Eigen::Matrix3d> e(priorWeight);
        Eigen::Matrix3d sqrtS = Eigen::Matrix3d::Identity();
        sqrtS.diagonal() = e.eigenvalues().real().array().sqrt();
        const Eigen::Matrix3d V = e.eigenvectors().real();
        priorWeight = V * sqrtS * V.transpose();
        //cout << "_cov.inverse:\n" << _cov.inverse() << endl;
        //cout << "priorWeight*priorWeight:\n"
        //     << priorWeight * priorWeight << endl;
        //cout << "priorWeight:\n" << priorWeight << endl;

        // 利用历史更新基线长度来提升先验权重，以期降低不确定度？？？？
#if ExpandPriorWeightAccordingToAccumulateBaseline
        //const double flatRatio = accumulateBaseline / lastUpdateAccBaseline * updateTime; // 下降过快
        //const double flatRatio = accumulateBaseline / lastUpdateAccBaseline; // 下降过慢

        // 计算历史更新应有权重
        // n = SW.size(), N = SW.size()+updateTime, flatratio = N/n = 1+updateTime/SW.size()
        const double flatRatio =
            1.0 + double(updateTime) / slidingWindow.size();  // 下降适中
        priorWeight *= flatRatio;
#endif
    }

    priorWeight.setZero();  // 不添加先验约束时使用

    vector<Eigen::Matrix3d> Rc_ws;
    vector<Eigen::Vector3d> Pc_ws;
    vector<vector<Eigen::Vector2d>> obvs;
    ExtrackPoseAndObvFromSlidingWindow(slidingWindow, Rc_ws, Pc_ws, obvs);
    SolveLandmarkPosition slp(Rc_ws, Pc_ws, obvs, K);
    slp.SetPriorPw(priorPw);
    slp.SetPriorWeight(priorWeight);

    optPw = slp.Optimize();

#define CalculateCovarianceByCeres 0

#if !CalculateCovarianceByCeres
    const Eigen::Matrix3d& H = CalculateHessianMatrix(slidingWindow, K, optPw);
    if (CalculateCovariance(H, cov, 1.0)) {
        return true;
    } else {
        return false;
    }
#else
    Eigen::Matrix<double, 3, 3, Eigen::RowMajor> covMatrix;
    if (slp.EstimateCovariance(covMatrix)) {
        cov = covMatrix;
        return true;
    } else {
        return false;
    }
    //const Eigen::Vector3d initPw = slp.GetPriorPw(); // 由历史值读取即可
#endif
}
