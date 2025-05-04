#include <deque>
#include <random>

#include "ceres_problem.h"
#include "classes.h"
#include "common.hpp"

using namespace std;

constexpr double huberTH = 5.99;
constexpr double noiseStd = 3.0;
constexpr double noiseStd2 = noiseStd * noiseStd;
constexpr double INF = 1e6;
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
    //constexpr int X0 = 220, Y0 = 200; // 这个可能导致位置差异太大，导致无法收敛？测试显示并不是，那是为啥呢？
    // 原因应该是X、Y值差异过小，但又是为什么要这样呢？它们相对位置都差不多啊？
    // 解答：问题最终定位为运动的姿态角有关系，不同姿态角导致三角化初值误差很大，进而影响优化算法无法收敛，
    // 飞行器的姿态角变换不可能很大，所以我们最终需要有一个方向正确的先验！！！
    // 前面描述的都不对：真正的原因是三角化时没有把Z>0这个条件用进去，
    // 不对这也只是其中一个原因而已 {400, 330}越靠近中心越不行，这是为什么呢？
    // 目前的解决方案是基于对地高度来给出initPw估计
    int X0 = 120, Y0 = 200;  // 为什么偏离图像中心反而可以
    // 设置一个平面点分布
    double radius = 3.0;  // 这个设置得>=1.0就不行了，但是增加采样点数量就又行了
    // 所以结论：
    // 1. 和半径大小有关
    // 2. 和采样点数量有关
    // 3. 和运动方式有关,最终影响投影点的位置
    // 4. 和噪声大小有关
    // 5. 最终影响先验估计值
    // 6. SVD求解Ax=0过程中，+X与-X都是解，我们要保证取+X
    // 设置一个在首个相机系下的深度
    double depth = 50.0;  // 认为是对地高度

    printf("Default parameters:\n\tX0=%d, Y0=%d, radius=%f, depth=%f\n", X0, Y0,
           radius, depth);
    if (argc > 2) {
        X0 = atoi(argv[1]);
        Y0 = atoi(argv[2]);
    }
    if (argc > 3) {
        radius = atof(argv[3]);
    }
    if (argc > 4) {
        depth = atof(argv[4]);
    }
    printf("Current parameters:\n\tX0=%d, Y0=%d, radius=%f,  depth=%f\n", X0,
           Y0, radius, depth);

    const Eigen::Vector3d Pw0 = depth * invK * Eigen::Vector3d(X0, Y0, 1);

    // 设置其它点的方向向量
    vector<Eigen::Vector3d> directions = {{1, 1, 0},   {-1, 1, 0}, {-1, -1, 0},
                                          {1, -1, 0},  {1, 3, 0},  {-1, 2, 0},
                                          {-1, -4, 0}, {1, 4, 0}};
    //vector<Eigen::Vector3d> directions = {
    //    {1, 1, 0},
    //    {-1, 1, 0},
    //    {-1, -1, 0},
    //    {1, -1, 0},
    //};
    if (noiseStd < 1.0) {
        directions = {};  // 不使用位置一致性
    }
    cout << "noiseStd: " << noiseStd << endl;

    for (Eigen::Vector3d& dir : directions) {
        dir.normalize();
    }

    // 求取仿真同平面世界点坐标
    Eigen::MatrixXd Pws(3, directions.size() + 1);
    Pws.col(0) = Pw0;
    for (int i = 1; i < Pws.cols(); ++i) {
        Pws.col(i) = Pw0 + radius * directions[i - 1];
    }
    cout << "Pws:\n" << Pws << endl;
    // 统计一下均值和方差
    const Eigen::Vector3d sumPw = Pws.rowwise().sum();
    const Eigen::Vector3d meanPw = sumPw / Pws.cols();
    cout << "sumPw: " << sumPw.transpose() << endl;
    cout << "meanPw: " << meanPw.transpose() << endl;
    const Eigen::MatrixXd normPws = Pws.colwise() - meanPw;
    cout << "normPws:\n" << normPws << endl;
    // 协方差是正定对称矩阵
    const Eigen::Matrix3d covPws = normPws * normPws.transpose();
    cout << "covPws:\n" << covPws.diagonal().array().transpose() << endl;

    // TODO: 分析点在归一化平面上的分布：Deepseek建议???

    // 初始化世界系的位置
    Eigen::Matrix3d lastRw_c = Eigen::Matrix3d::Identity();
    Eigen::Vector3d lastPw_c = Eigen::Vector3d::Zero();

    // 历史位置估计值及方差

    PriorEstimateData priorData;

    // 滑动窗口
    deque<DataFrame> slidingWindow;
    double accumulateBaseline = 0.0;
    double lastUpdateAccBaseline = 0.0;

    // 设置运动参数
    constexpr int maxUpdateTime = 100;
    const Eigen::Vector2d rotRange(-0.05, 0.1);  // degree
    // TODO: 实验显示，帧间距离越近，不确定度会非常大
    //const Eigen::Vector2d moveRange(0.01, 0.1);  // meter
    // 帧间基线越大，J矩阵值越大(信息越大)，协方差下降越快
    const Eigen::Vector2d moveRange(-0.05, 0.1);  // meter

    bool isInitialized = false;

    int updateTime = 0;
    while (updateTime < maxUpdateTime) {
        cout << endl;
        //random_device rd; // 不使用真随机数
        mt19937 gen1(42), gen2(43);
        uniform_real_distribution<double> rd(rotRange.x(), rotRange.y());
        uniform_real_distribution<double> md(moveRange.x(), moveRange.y());

        // 生成旋转及平移大小
        const double rotAng = rd(gen1);
        const double moveDist = md(gen2);
        // 生成旋转及平移方向
        Eigen::Vector3d rotDir(0, 0, md(gen1));  // 主要绕Z轴，不然就翻车了
        //Eigen::Vector3d posDir(md(gen2) * 0.01, md(gen2) * 0.01,
        //                       md(gen2));  // 沿任何轴平移均可
        Eigen::Vector3d posDir(md(gen2), md(gen2), 0);  // 不沿Z轴

        // 产生下一个位姿
        Eigen::Matrix3d Rw_c2;
        Eigen::Vector3d Pw_c2;
        GenerateNextPose(lastRw_c, lastPw_c, rotDir, posDir, rotAng, moveDist,
                         Rw_c2, Pw_c2);
        if (Pw_c2.z() >= depth - 5.0) {
            Pw_c2.z() = depth - 5.0;  // 避免观测不到marker点
        }

        if (!slidingWindow.empty()) {
            double moveDist = (lastPw_c - Pw_c2).norm();
            accumulateBaseline += moveDist;
        }
        lastRw_c = Rw_c2;
        lastPw_c = Pw_c2;

        cout << "current accumulateBaseline: " << accumulateBaseline << endl;

        // 计算投影观测，并产生一帧DataFrame
        vector<Eigen::Vector2d> obvEachFrame;
        mt19937 gen(44);
        normal_distribution<double> dist(0.0, noiseStd);
        // 遍历每一个世界点，将其投影到当前帧
        const Eigen::Matrix3d& Rc_w = Rw_c2.transpose();
        const Eigen::Vector3d& Pc_w =
            Rc_w * (-Pw_c2);  // 先将相对向量转向，再转到另一个坐标系下

        cv::Mat debugImg(kImgHeight, kImgWidth, CV_8UC3);
        debugImg.setTo(cv::Scalar(0, 0, 0));
        // 遍历每一个世界点，计算投影点
        for (int j = 0; j < Pws.cols(); ++j) {
            const Eigen::Vector3d& Pw = Pws.col(j);
            // 这里其实也要考虑位姿的扰动误差
            Eigen::Vector2d _obv = ProjectPw2Pixel(Pw, Rc_w, Pc_w, K);
            _obv[0] = int(_obv[0]);
            _obv[1] = int(_obv[1]);

            const Eigen::Vector2d noise{int(dist(gen)), int(dist(gen))};
            const Eigen::Vector2d _obv_noisy = _obv + noise;
            obvEachFrame.push_back(_obv_noisy);

            //cout << "noise: " << noise.transpose() << endl;
            //cout << "obv: " << _obv.transpose() << endl;

            cv::circle(debugImg, cv::Point2i(_obv[0], _obv[1]), 1,
                       cv::Scalar(0, 255, 0), -1);
            cv::circle(debugImg, cv::Point2i(_obv_noisy[0], _obv_noisy[1]), 1,
                       {0, 0, 255}, -1);
        }

        // 位姿添加到滑窗
        DataFrame frame(Rw_c2.transpose(), Rw_c2.transpose() * -Pw_c2, depth,
                        obvEachFrame, double(updateTime), debugImg);
        slidingWindow.push_back(frame);

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
            3.0 * noiseStd;  // meters，条件数越小，这个相对调大
        // 深度为dm，归一化坐标为(xn, yn, 1)，那么(Xw, Yw) = (d*xn, d*yn)，
        // dm=dt+Δd，那么(Xw, Yw)的误差为(Δd*xn, Δd*yn)，易知，(xn, yn)绝对值不超过，
        // 并且：
        // 1. 像素越靠近中心，(xn, yn)值越小，因此误差Δd引起的(ΔXw, ΔYw)越小；
        // 2. 像素越远离中心，(xn, yn)虽然值越大，但是深度的辨识系数越显著，深度不确定度Δd越小，Δd引起的(ΔXw, ΔYw)越小
        // 我们更关心水平位置精度，因此这里只关心水平位置精度即可
        const Eigen::Vector3d& std = cov.diagonal().array().sqrt().transpose();
        if (std.head(2).norm() > maxMakerPosStd) {
            cout << "std too big, value: " << std.transpose() << endl;
            CullingBadObservationsBeforeInit(slidingWindow);
            continue;
        }

        isInitialized = true;
        priorData.UpdateMeanAndH(optPw, cov);

        printf("accBaseline - lastAcc: %f-%f=%f\nSW size=%ld\n",
               accumulateBaseline, lastUpdateAccBaseline,
               (accumulateBaseline - lastUpdateAccBaseline),
               slidingWindow.size());
        lastUpdateAccBaseline = accumulateBaseline;
        ++updateTime;

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

            // 2、再指定残差块，这样就不需要再后续指定problem.SetManifold(q,
            // quaternion_parameterization)
            // 距离越远，容忍的像素误差越小
            ceres::LossFunction* huber = new ceres::HuberLoss(huberTH);
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
    options.logging_type =
        ceres::SILENT;  // PER_MINIMIZER_ITERATION;  // 设置输出log便于bug排查
    options.function_tolerance = 1e-12;
    options.inner_iteration_tolerance = 1e-6;

    // 运行优化
    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem_, &summary);
    return optPw_;
}

bool SolveLandmarkPosition::EstimateCovariance(
    Eigen::Matrix<double, 3, 3, Eigen::RowMajor>& covMatrix) {
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

    covMatrix.setConstant(INF);
    if (!covariance.Compute({optPw_.data()}, &problem_)) {
        cerr << "Failed to compute covariance！！！" << endl;
        cout << "Failed to compute covariance！！！" << endl;
        return false;
    } else {
        covariance.GetCovarianceBlock(optPw_.data(), optPw_.data(),
                                      covMatrix.data());
        covMatrix *= noiseStd2;
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
    Eigen::Matrix<double, 3, 3, Eigen::RowMajor> covMatrix;
    if (slp.EstimateCovariance(covMatrix)) {
        cov = covMatrix;
        return true;
    } else {
        return false;
    }
    //const Eigen::Vector3d initPw = slp.GetPriorPw(); // 由历史值读取即可
}
