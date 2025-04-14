#include "common.hpp"

using namespace std;

#define USE_AUTO_DIFF 0

constexpr double huberTH = 5.99;
constexpr double noiseStd = 3.0;
constexpr double noiseStd2 = noiseStd * noiseStd;
constexpr double INF = 1e20;
constexpr int ceresMaxIterativeTime = 1000;

// 每次量测更新时所需要的最少帧数量，比这个更重要的是基线长度
constexpr int kMinUpdateFrameNumEachTime = 10;
// 姿态、观测条件都一样的情况下，最影响地图点位置不确定度的只有基线
// 其次就是量测的个数了，如果基线长短不断变换，那么标准差自然来回波动，
// 如果累积基线长度不断增大，那么标准差自然不断减小，
// 问题是：如何把先验转换为累积基线，以降低标准差
constexpr double kMinUpdateBaselineInSW = 6.5;
// 一个可行的办法是增大先验的权重，这样就能每次降低不确定度，且必须增长够快，如线性增长
// 才能使std不断下降，这是因为W增大，海森矩阵H增大，而最终的方差与H矩阵的逆正相关
#define ExpandPriorWeightAccordingToAccumulateBaseline 1

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

class PriorPwResidual : public ceres::CostFunction {
   public:
    PriorPwResidual(const Eigen::Vector3d& p, const Eigen::Matrix3d& w)
        : priorPw_(p), weight_(w) {
        set_num_residuals(3);
        mutable_parameter_block_sizes()->push_back(3);
    }
    virtual bool Evaluate(double const* const* parameters, double* residuals,
                          double** jacobians) const override;

   private:
    Eigen::Vector3d priorPw_;
    Eigen::Matrix3d weight_;
};
#endif

Eigen::Matrix3d RPY2Rotation(const Eigen::Vector3d& _rpy,
                             const bool isDeg = true);

Eigen::Vector2d ProjectPw2Pixel(const Eigen::Vector3d& Pw,
                                const Eigen::Matrix3d& Rcw,
                                const Eigen::Vector3d& Pcw);

// 实际实验发现，只要方向对就行，尽管初值很不准，但在所估计的初值方向附近
// 只有最优值这么一个极小值，去验证吧！
Eigen::Vector3d EstimatePwInitialValue(
    const vector<Eigen::Matrix3d>& Rcw, const vector<Eigen::Vector3d>& Pcw,
    const vector<vector<Eigen::Vector2d>>& obvs, const ::Eigen::Matrix3d& K);

Eigen::Vector3d EstimatePwInitialValueNormlized(
    const vector<Eigen::Matrix3d>& Rcw, const vector<Eigen::Vector3d>& Pcw,
    const vector<vector<Eigen::Vector2d>>& obvs, const ::Eigen::Matrix3d& K);

class SolveLandmarkPosition {
   public:
    SolveLandmarkPosition(const vector<Eigen::Matrix3d>& Rc_w,
                          const vector<Eigen::Vector3d>& Pc_w,
                          const vector<vector<Eigen::Vector2d>>& obvs,
                          const Eigen::Matrix3d& _K,
                          const Eigen::Vector3d& priorPw = {0., 0., 0.});
    SolveLandmarkPosition() = delete;
    Eigen::Vector3d Optimize();
    Eigen::Matrix3d EstimateCovariance();
    Eigen::Vector3d GetPriorPw() const { return priorPw_; }
    void SetPriorPw(const Eigen::Vector3d& p) { priorPw_ = p; }
    void SetPriorWeight(const Eigen::Matrix3d& w) { priorWeight_ = w; }

   private:
    vector<Eigen::Matrix3d> Rc_w_;
    vector<Eigen::Vector3d> Pc_w_;
    vector<vector<Eigen::Vector2d>> obvs_;
    Eigen::Vector3d priorPw_;  // 待估计变量
    Eigen::Matrix3d priorWeight_ = Eigen::Matrix3d::Zero();
    Eigen::Vector3d optPw_;
    ceres::Problem problem_;
    Eigen::Matrix3d K_;
};

/**
 * @brief 拼接多张图像并绘制观测点匹配连线
 * @param debugImgs 输入的图像列表（每个相机视角一张图）
 * @param obvs 观测点列表（每个视角对应的观测像素坐标）
 * @return cv::Mat 拼接后的图像
 */
cv::Mat stitchAndDrawMatches(
    const std::deque<cv::Mat>& debugImgs,
    const std::deque<std::vector<Eigen::Vector2d>>& obvs);

// 直接利用对地高度计算初值
Eigen::Vector3d CalculatePriorPwByHeight(const Eigen::Matrix3d& Rwc,
                                         const Eigen::Vector3d& Pwc,
                                         const double& horizontalHeight,
                                         const Eigen::Matrix3d& K,
                                         const vector<Eigen::Vector2d>& obvs);

bool SlidingWindowSolvedByCeres(const deque<DataFrame>& slidingWindow,
                                const vector<Eigen::Vector3d>& historyEstPw,
                                const vector<Eigen::Matrix3d>& historyEstCov,
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
    bool usePrior = false;

    printf(
        "Default parameters:\n\tX0=%d, Y0=%d, radius=%f, depth=%f, "
        "usePrior=%d,\n",
        X0, Y0, radius, depth, usePrior);
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
    if (argc > 5) {
        usePrior = bool(atoi(argv[5]));
    }
    printf(
        "Current parameters:\n\tX0=%d, Y0=%d, radius=%f,  depth=%f, "
        "usePrior=%d\n",
        X0, Y0, radius, depth, usePrior);

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
    vector<Eigen::Vector3d> historyEstPw;
    vector<Eigen::Matrix3d> historyEstCov;

    // 滑动窗口
    deque<DataFrame> slidingWindow;
    double accumulateBaseline = 0.0;
    double lastUpdateAccBaseline = 0.0;
    deque<cv::Mat> debugImgs;
    deque<vector<Eigen::Vector2d>> debugObvs;

    // 设置运动参数
    constexpr int maxUpdateTime = 10;
    const Eigen::Vector2d rotRange(0.05, 0.1);  // degree
    // TODO: 实验显示，帧间距离越近，不确定度会非常大
    //const Eigen::Vector2d moveRange(0.01, 0.1);  // meter
    // 帧间基线越大，J矩阵值越大(信息越大)，协方差下降越快
    const Eigen::Vector2d moveRange(0.5, 1.0);  // meter

    int updateTime = 0;
    while (updateTime < maxUpdateTime) {
        //random_device rd; // 不使用真随机数
        mt19937 gen1(42), gen2(43);
        uniform_real_distribution<double> rd(rotRange.x(), rotRange.y());
        uniform_real_distribution<double> md(moveRange.x(), moveRange.y());

        // 生成旋转及平移大小
        const double rotAng = rd(gen1);
        const double moveDist = md(gen2);
        // 生成旋转及平移方向
        Eigen::Vector3d rotDir(0, 0, 1);  // 主要绕Z轴，不然就翻车了
        Eigen::Vector3d posDir(md(gen2), md(gen2),
                               md(gen2));  // 沿任何轴平移均可

        // 产生下一个位姿
        Eigen::Matrix3d Rw_c2;
        Eigen::Vector3d Pw_c2;
        GenerateNextPose(lastRw_c, lastPw_c, rotDir, posDir, rotAng, moveDist,
                         Rw_c2, Pw_c2);
        lastRw_c = Rw_c2;
        lastPw_c = Pw_c2;
        if (!slidingWindow.empty()) {
            const DataFrame& f = slidingWindow.back();
            double moveDist =
                (-f.Rc_w.transpose() * f.Pc_w + Rw_c2.transpose() * Pw_c2)
                    .norm();
            accumulateBaseline += moveDist;
        }

        // 计算投影观测，并产生一帧DataFrame
        vector<Eigen::Vector2d> obvs;
        mt19937 gen(44);
        cout << "noiseStd: " << noiseStd << endl;
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
            const Eigen::Vector2d _obv = ProjectPw2Pixel(Pw, Rc_w, Pc_w);

            const Eigen::Vector2d noise{dist(gen), dist(gen)};
            const Eigen::Vector2d _obv_noisy = _obv + noise;
            obvs.push_back(_obv_noisy);

            //cout << "noise: " << noise.transpose() << endl;
            //cout << "obv: " << _obv.transpose() << endl;

            cv::circle(debugImg, cv::Point2i(_obv[0], _obv[1]), 1,
                       cv::Scalar(0, 255, 0), -1);
            cv::circle(debugImg, cv::Point2i(_obv_noisy[0], _obv_noisy[1]), 1,
                       {0, 0, 255}, -1);
        }

        // 位姿添加到滑窗
        DataFrame frame(Rw_c2.transpose(), Rw_c2.transpose() * -Pw_c2, depth,
                        obvs, double(updateTime));
        slidingWindow.push_back(frame);
        debugObvs.push_back(obvs);
        debugImgs.push_back(debugImg);

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
        const cv::Mat mergeImg = stitchAndDrawMatches(debugImgs, debugObvs);
        cv::imshow("mergeImg_X0:" + to_string(X0) + "_Y0:" + to_string(Y0),
                   mergeImg);
        cv::waitKey(0);

        // 执行一次滑窗优化，并弹出最老一帧
        Eigen::Vector3d optPw;
        Eigen::Matrix3d cov;
        SlidingWindowSolvedByCeres(slidingWindow, historyEstPw, historyEstCov,
                                   accumulateBaseline, lastUpdateAccBaseline,
                                   updateTime, optPw, cov);
        historyEstPw.push_back(optPw);
        historyEstCov.push_back(cov);
        printf("accBaseline - lastAcc: %f-%f=%f\nSW size=%ld\n",
               accumulateBaseline, lastUpdateAccBaseline,
               (accumulateBaseline - lastUpdateAccBaseline),
               slidingWindow.size());
        lastUpdateAccBaseline = accumulateBaseline;
        ++updateTime;

        // 移除滑动窗口元素
        auto MoveSlidingWindow = [&slidingWindow, &debugImgs,
                                  &debugObvs]() -> void {
            slidingWindow.pop_front();
            debugImgs.pop_front();
            debugObvs.pop_front();
        };
        //while (slidingWindow.size() >= kMinUpdateFrameNumEachTime) {
        //    MoveSlidingWindow();
        //}
        // 仅移动一个
        MoveSlidingWindow();

        // 打印输出
        //cout << "initPw: " << initPw.transpose() << endl;
        cout << "optPw: " << optPw.transpose() << endl;
        cout << "std: " << cov.diagonal().array().sqrt().transpose() << endl
             << endl;
    }

    /*
    // 仿真观测到上述地图点的相机位姿
    const vector<Eigen::Vector3d> vecAngleRwc{
        //{1, 2, 5}, {0, 10, 10}, {20, 3, 15}};
        {1, 2, 5},
        {0, 1, 10},
        {2, 3,
         15}};  // 这个初始设定在X0 = 220, Y0 = 200或X0=100, Y0 = 150时初始值估计误差很大，导致结果无法收敛
    const vector<Eigen::Vector3d> vecPw{{2, 3, 4}, {3, 4, 7}, {4, 3, 8}};

    vector<Eigen::Matrix3d> Rc_w;
    for (int i = 0; i < vecAngleRwc.size(); ++i) {
        Rc_w.push_back(RPY2Rotation(vecAngleRwc[i]).transpose());
    }
    vector<Eigen::Vector3d> Pc_w;
    for (int i = 0; i < vecPw.size(); ++i) {
        Pc_w.push_back(-Rc_w[i] * vecPw[i]);
    }

    constexpr int loopTime = 1;  // 保留历史位姿，并使用EKF进行位置更新
    vector<Eigen::Vector3d> historyOptPw;
    vector<Eigen::Vector3d> historyInitPw;
    vector<Eigen::Matrix3d> historyCov;
    for (int t = 0; t < loopTime; ++t) {

        vector<vector<Eigen::Vector2d>> obvs(vecPw.size());
        vector<cv::Mat> debugImg(vecPw.size());

        random_device rd;
        mt19937 gen(rd());
        cout << "noiseStd: " << noiseStd << endl;
        normal_distribution<double> dist(0.0, noiseStd);
        //cv::Mat img(kImgHeight, kImgWidth, CV_8UC3, {0, 0, 0});
        // 产生一些图像观测，并添加噪声
        for (int i = 0; i < vecPw.size(); ++i) {
            debugImg[i].create(kImgHeight, kImgWidth, CV_8UC3);
            debugImg[i].setTo(cv::Scalar(0, 0, 0));
            cv::Mat img = debugImg[i];

            // 遍历每一个世界点，将其投影到当前帧
            for (int j = 0; j < Pws.cols(); ++j) {
                const Eigen::Vector3d& Pw = Pws.col(j);
                // 这里其实也要考虑位姿的扰动误差
                const Eigen::Vector2d _obv =
                    ProjectPw2Pixel(Pw, Rc_w[i], Pc_w[i]);

                const Eigen::Vector2d noise{dist(gen), dist(gen)};
                const Eigen::Vector2d _obv_noisy = _obv + noise;
                obvs[i].push_back(_obv_noisy);

                //cout << "noise: " << noise.transpose() << endl;
                //cout << "obv: " << _obv.transpose() << endl;

                cv::circle(img, cv::Point2i(_obv[0], _obv[1]), 1,
                           cv::Scalar(0, 255, 0), -1);
                cv::circle(img, cv::Point2i(_obv_noisy[0], _obv_noisy[1]), 1,
                           {0, 0, 255}, -1);
            }
        }

        for (int k = 0; k < debugImg.size(); ++k) {
            cv::imshow("img_" + to_string(k), debugImg[k]);
        }
        cv::waitKey(0);

        const cv::Mat mergeImg = stitchAndDrawMatches(debugImg, obvs);
        cv::imshow("mergeImg_X0:" + to_string(X0) + "_Y0:" + to_string(Y0),
                   mergeImg);
        cv::waitKey(0);

        // 利用对地高度计算初值，因为X logo被认为是在地面即d=0，但现在让其为depth=50，
        // 因此我参考系也要往上平移50
        Eigen::Vector3d priorPw = Eigen::Vector3d::Zero();
        for (int i = 0; i < Pc_w.size(); ++i) {
            priorPw += CalculatePriorPwByHeight(Rc_w[i].transpose(),
                                                -Rc_w[i].transpose() * Pc_w[i],
                                                depth, K, obvs[i]);
        }
        priorPw /= Pc_w.size();
        cout << "CalculatePriorPwByHeight: " << priorPw.transpose() << endl;

        SolveLandmarkPosition slp(Rc_w, Pc_w, obvs, K);
        if (usePrior) {
            // 给一个很不准，但是方向正确的试试，
            // Shame，无法得到正确结果
            slp.SetPriorPw(priorPw);
        }
        const Eigen::Vector3d optPw = slp.Optimize();
        const Eigen::Matrix3d cov = slp.EstimateCovariance();
        const Eigen::Vector3d initPw = slp.GetPriorPw();
        historyInitPw.emplace_back(initPw);

        cout << "initPw: " << initPw.transpose() << endl;
        cout << "optPw: " << optPw.transpose() << endl;
        cout << "std: " << cov.diagonal().array().sqrt().transpose() << endl
             << endl;
    }
*/
    /*
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
*/
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

Eigen::Vector3d EstimatePwInitialValue(
    const vector<Eigen::Matrix3d>& Rcw, const vector<Eigen::Vector3d>& Pcw,
    const vector<vector<Eigen::Vector2d>>& obvs, const Eigen::Matrix3d& K) {
    // px2 = ρ2 * K * [Rcw, Pcw] * Pw
    // [px2]x * px2 = [px2]x * K * [Rcw, Pcw] * Pw = 0
    Eigen::MatrixXd A;
    A.resize(3 * Rcw.size() * obvs[0].size(), 4);
    int rowId = 0;
    for (int i = 0; i < Rcw.size(); ++i) {
        // 遍历当前帧能够观测到的所有地图点
        for (int j = 0; j < obvs[i].size(); ++j) {
            const Eigen::Vector3d obv_i{obvs[i][j](0), obvs[i][j](1), 1.0};
            Eigen::Matrix<double, 3, 4> T;
            T.block(0, 0, 3, 3) = Rcw[i];
            T.block(0, 3, 3, 1) = Pcw[i];
            A.block(rowId, 0, 3, 4) = skew(obv_i) * K * T;
            rowId += 3;
        }
    }
    Eigen::JacobiSVD<Eigen::MatrixXd> svd(A, Eigen::ComputeFullV);
    const Eigen::Vector4d bestV =
        svd.matrixV().col(svd.singularValues().size() - 1);
    const Eigen::Vector3d estPw = bestV.head(3) / bestV[3];
    // cout << "A:\n" << A << endl;
    // 这里有一个0空间，若要有解，则最后一个特征值要极小
    cout << "singularValues: " << svd.singularValues().transpose() << endl;
    // cout << "est Pw: " << estPw.transpose() << endl;
    if (estPw.z() < 0) {
        // 原因是这里有时候会解算出负值，根据Ax=0，x的正负不影响结果，
        // 因此需要加一个先验判断
        // 但是也存在着这里解算不准的情况，因此需要加一个由对地高度直接解算位置的先验
        return -estPw;
        ;
    }
    return estPw;
}

Eigen::Vector3d EstimatePwInitialValueNormlized(
    const vector<Eigen::Matrix3d>& Rcw, const vector<Eigen::Vector3d>& Pcw,
    const vector<vector<Eigen::Vector2d>>& obvs, const Eigen::Matrix3d& K) {
    // px2 = ρ2 * K * [Rcw, Pcw] * Pw
    // [px2]x * px2 = [px2]x * K * [Rcw, Pcw] * Pw = 0
    // 设归一化放缩矩阵为S，则有：
    // S * px2 = S * ρ2 * K * [Rcw, Pcw] * Pw
    // [S * px2]x * S * px2 = [S * px2]x * S * K * [Rcw, Pcw] * Pw = 0
    Eigen::MatrixXd A;
    A.resize(3 * Rcw.size() * obvs[0].size(), 4);
    int rowId = 0;
    for (int i = 0; i < Rcw.size(); ++i) {
        // 将所有观测放缩到以(0, 0)为中心，长为sqrt(2)的圆内
        const double sqrt2 = sqrt(2.0);
        Eigen::Vector2d meanObv = Eigen::Vector2d::Zero();
        vector<Eigen::Vector2d> meanObvs;
        for (int j = 0; j < obvs[i].size(); ++j) {
            meanObv += obvs[i][j];
        }
        meanObv /= obvs[i].size();

        // 遍历当前帧能够观测到的所有地图点
        const Eigen::Vector3d origin(meanObv[0], meanObv[1], 1);
        for (int j = 0; j < obvs[i].size(); ++j) {
            const Eigen::Vector3d obv_i{obvs[i][j](0), obvs[i][j](1), 1.0};
            // 计算当前观测的放缩系数及放缩矩阵
            const double scaleFactor = sqrt2 / (obv_i - origin).norm();
            Eigen::Matrix3d scaleMatrix =
                (Eigen::Matrix3d() << scaleFactor, 0, -scaleFactor * origin.x(),
                 0, scaleFactor, -scaleFactor * origin.y(), 0, 0, 1)
                    .finished();
            // 放缩观测
            const Eigen::Vector3d sObv_i = scaleMatrix * obv_i;
            Eigen::Matrix<double, 3, 4> T;
            T.block(0, 0, 3, 3) = Rcw[i];
            T.block(0, 3, 3, 1) = Pcw[i];
            A.block(rowId, 0, 3, 4) = skew(sObv_i) * scaleMatrix * K * T;
            rowId += 3;
        }
    }
    Eigen::JacobiSVD<Eigen::MatrixXd> svd(A, Eigen::ComputeFullV);
    const Eigen::Vector4d bestV =
        svd.matrixV().col(svd.singularValues().size() - 1);
    const Eigen::Vector3d estPw = bestV.head(3) / bestV[3];
    // cout << "A:\n" << A << endl;
    // 这里有一个0空间，若要有解，则最后一个特征值要极小
    cout << "singularValues normlized: " << svd.singularValues().transpose() << endl;
    // cout << "est Pw: " << estPw.transpose() << endl;
    if (estPw.z() < 0) {
        // 原因是这里有时候会解算出负值，根据Ax=0，x的正负不影响结果，
        // 因此需要加一个先验判断
        // 但是也存在着这里解算不准的情况，因此需要加一个由对地高度直接解算位置的先验
        return -estPw;
        ;
    }
    return estPw;
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
#if USE_AUTO_DIFF
            ceres::CostFunction* cost_function =
                new ceres::AutoDiffCostFunction<ProjectionResidual, 2, 3>(
                    new ProjectionResidual(Rc_w_[i], Pc_w_[i], obv, K_));
#else
            ceres::CostFunction* cost_function =
                new ProjectionResidual(Rc_w_[i], Pc_w_[i], obv, K_);
#endif
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
    //options.logging_type =
    //    ceres::PER_MINIMIZER_ITERATION;  // 设置输出log便于bug排查
    options.logging_type = ceres::SILENT;
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
    covMatrix.setConstant(INF);
    if (!covariance.Compute({optPw_.data()}, &problem_)) {
        cerr << "Failed to compute covariance！！！" << endl;
    } else {
        covariance.GetCovarianceBlock(optPw_.data(), optPw_.data(),
                                      covMatrix.data());

        // Output the covariance matrix
        //cout << "Covariance matrix for p:\n" << covMatrix << endl;
    }
    return noiseStd2 * covMatrix;
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

cv::Mat stitchAndDrawMatches(
    const std::deque<cv::Mat>& debugImgs,
    const std::deque<std::vector<Eigen::Vector2d>>& obvs) {
    if (debugImgs.empty() || obvs.empty()) {
        return cv::Mat();
    }

    // --- 1. 图像拼接 ---
    // 计算拼接后的总宽度和最大高度
    int totalWidth = 0;
    int maxHeight = 0;
    for (const auto& img : debugImgs) {
        totalWidth += img.cols;
        maxHeight = std::max(maxHeight, img.rows);
    }

    // 创建拼接画布
    cv::Mat canvas(maxHeight, totalWidth, CV_8UC3, cv::Scalar(0, 0, 0));

    // 将图像横向拼接
    int x_offset = 0;
    for (const auto& img : debugImgs) {
        cv::Mat roi = canvas(cv::Rect(x_offset, 0, img.cols, img.rows));
        img.copyTo(roi);
        x_offset += img.cols;
        // 这里画一个竖条纹，以区分不同帧
        cv::line(canvas, {x_offset, 0}, {x_offset, img.rows - 1},
                 {255, 255, 255}, 2);
    }

    // --- 2. 绘制观测点连线 ---
    const std::vector<cv::Scalar> colors = {
        cv::Scalar(0, 255, 0),    // 绿色
        cv::Scalar(0, 0, 255),    // 红色
        cv::Scalar(255, 0, 0),    // 蓝色
        cv::Scalar(255, 255, 0),  // 青色
        cv::Scalar(255, 0, 255),  // 紫色
        cv::Scalar(0, 255, 255)   // 黄色
    };

    // 遍历每个地图点（假设所有视角观测到相同数量的点）
    for (size_t pt_idx = 0; pt_idx < obvs[0].size(); ++pt_idx) {
        cv::Scalar color = colors[pt_idx % colors.size()];

        // 存储当前点在所有视角中的位置
        std::vector<cv::Point> points;
        int x_accum = 0;

        // 遍历每个视角
        for (size_t view_idx = 0; view_idx < obvs.size(); ++view_idx) {
            if (pt_idx >= obvs[view_idx].size())
                continue;

            // 计算当前点在拼接图中的绝对坐标
            const Eigen::Vector2d& obv = obvs[view_idx][pt_idx];
            cv::Point pt(x_accum + obv.x(), obv.y());
            points.push_back(pt);

            // 绘制当前视角的观测点
            cv::circle(canvas, pt, 3, color, -1);

            // 如果是第一个视角，跳过连线
            if (view_idx == 0) {
                x_accum += debugImgs[view_idx].cols;
                continue;
            }

            // 绘制与前一视角的连线
            cv::line(canvas, points[view_idx - 1], points[view_idx], color, 1);
            x_accum += debugImgs[view_idx].cols;
        }
    }

    return canvas;
}

Eigen::Vector3d CalculatePriorPwByHeight(const Eigen::Matrix3d& Rwc,
                                         const Eigen::Vector3d& Pwc,
                                         const double& horizontalHeight,
                                         const Eigen::Matrix3d& K,
                                         const vector<Eigen::Vector2d>& obvs) {
    Eigen::Vector3d sumPw = Eigen::Vector3d::Zero();
    // 相机在W下表示的与X logo所在水平面的距离，
    // 这里是由相机->水平面的向量
    const double dw = horizontalHeight - Pwc.z();
    const Eigen::Vector3d Pw(0, 0, dw);

    // 将W系下表示的距离变换到C系下
    const double dc = Rwc.transpose().row(2)[2] * dw;
    printf("dw=%f, dc=%f\n", dw, dc);

    // 先计算在相机系下的坐标，再转到世界系
    for (const Eigen::Vector2d& px : obvs) {
        const Eigen::Vector3d Pc =
            //dc * invK * Eigen::Vector3d(px[0], px[1], 1.0);
            dw * invK * Eigen::Vector3d(px[0], px[1], 1.0);

        // 转换到世界系下的位置
        sumPw += Rwc * Pc;
    }

    return sumPw / obvs.size();
}

bool PriorPwResidual::Evaluate(double const* const* parameters,
                               double* residuals, double** jacobians) const {
    const double* p = parameters[0];
    const Eigen::Vector3d optPw(p[0], p[1], p[2]);

    // 计算残差
    const Eigen::Vector3d res = weight_ * (optPw - priorPw_);
    residuals[0] = res[0];
    residuals[1] = res[1];
    residuals[2] = res[2];

    if (jacobians) {
        if (jacobians[0]) {
            Eigen::Map<Eigen::Matrix<double, 3, 3, Eigen::RowMajor>> j(
                jacobians[0]);
            j = weight_;
        }
    }

    return true;
}

bool SlidingWindowSolvedByCeres(const deque<DataFrame>& slidingWindow,
                                const vector<Eigen::Vector3d>& historyEstPw,
                                const vector<Eigen::Matrix3d>& historyEstCov,
                                const double& accumulateBaseline,
                                const double& lastUpdateAccBaseline,
                                const int updateTime, Eigen::Vector3d& optPw,
                                Eigen::Matrix3d& cov) {
    vector<Eigen::Matrix3d> Rc_w(slidingWindow.size());
    vector<Eigen::Vector3d> Pc_w(slidingWindow.size());
    vector<double> height2Ground(slidingWindow.size());
    vector<vector<Eigen::Vector2d>> obvs(slidingWindow.size());
    for (int i = 0; i < slidingWindow.size(); ++i) {
        const DataFrame& frame = slidingWindow[i];
        Rc_w[i] = frame.Rc_w;
        Pc_w[i] = frame.Pc_w;
        height2Ground[i] = frame.height2Ground;
        obvs[i] = frame.obv;
    }
    // 利用对地高度计算初值，因为X logo被认为是在地面即d=0，但现在让其为depth=50，
    // 因此我参考系也要往上平移50
    Eigen::Vector3d priorPw = Eigen::Vector3d::Zero();
    Eigen::Matrix3d priorWeight = Eigen::Matrix3d::Zero();
    if (historyEstPw.empty()) {
#define CalculateInitPwByDLT 1

#if !CalculateInitPwByDLT
        for (int i = 0; i < Pc_w.size(); ++i) {
            cout << "Pw_c[" << i
                 << "]: " << (-Rc_w[i].transpose() * Pc_w[i]).transpose()
                 << endl;
            const double& depth = height2Ground[i];
            priorPw += CalculatePriorPwByHeight(Rc_w[i].transpose(),
                                                -Rc_w[i].transpose() * Pc_w[i],
                                                depth, K, obvs[i]);
        }
        priorPw /= Pc_w.size();
        cout << "CalculatePriorPwByHeight: " << priorPw.transpose() << endl;
#else 
    // 使用DLT计算初始值
    const Eigen::Vector3d initPw1 = EstimatePwInitialValue(Rc_w, Pc_w, obvs, K);
    const Eigen::Vector3d initPw2 = EstimatePwInitialValueNormlized(Rc_w, Pc_w, obvs, K);
    cout << "initPw1: " << initPw1.transpose() << endl;
    cout << "initPw2: " << initPw2.transpose() << endl;
    priorPw = initPw1;
#endif

    } else {
        priorPw = historyEstPw.back();
        Eigen::Matrix3d _cov = historyEstCov.back();
        constexpr double epsilon = 1e-10;
        _cov.diagonal() += Eigen::Vector3d::Ones() * epsilon;
        // 计算信息矩阵的开根号
        Eigen::EigenSolver<Eigen::Matrix3d> e(_cov.inverse());
        Eigen::Matrix3d sqrtS = Eigen::Matrix3d::Identity();
        sqrtS.diagonal() = e.eigenvalues().real().array().sqrt();
        const Eigen::Matrix3d V = e.eigenvectors().real();
        priorWeight = V * sqrtS * V.transpose();
        cout << "_cov.inverse:\n" << _cov.inverse() << endl;
        cout << "priorWeight*priorWeight:\n"
             << priorWeight * priorWeight << endl;
        cout << "priorWeight:\n" << priorWeight << endl;

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
    SolveLandmarkPosition slp(Rc_w, Pc_w, obvs, K);
    slp.SetPriorPw(priorPw);
    slp.SetPriorWeight(priorWeight);

    optPw = slp.Optimize();
    cov = slp.EstimateCovariance();
    //const Eigen::Vector3d initPw = slp.GetPriorPw(); // 由历史值读取即可
    return true;
}
