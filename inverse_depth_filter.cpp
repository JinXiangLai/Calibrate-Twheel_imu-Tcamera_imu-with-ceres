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
constexpr double kZeroLine = 1e-10;

int main(int argc, char** argv) {
    double radius = 3.0;        // 每个marker半径
    double depth = 50.0;        //
    double distX = radius * 4;  // 两个marker距离
    constexpr int kMarkerNum = 2;
    double startX = 10, startY = 11;
    printf("Default parameters:\n\tradius=%f, depth=%f, distX=%f\n", radius,
           depth, distX);
    if (argc > 1) {
        radius = atof(argv[1]);
        distX = radius * 4;
    }
    if (argc > 2) {
        depth = atof(argv[2]);
    }
    if (argc > 3) {
        startX = atof(argv[3]);
    }
    if (argc > 4) {
        startY = atof(argv[4]);
    }
    printf("Current parameters:\n\tradius=%f, depth=%f, distX=%f, startX=%f, startY=%f\n", radius,
           depth, distX, startX, startY);

    // 初始化两个一模一样的marker点
    // clang-format off
    Eigen::Matrix<double, 3, kMarkerNum> Pws = 
        (Eigen::Matrix<double, 3, kMarkerNum>() << startX, startX + distX,
                                                   startY, startY,
                                                   depth, depth).finished();
    // clang-format on
    cout << "Pws:\n" << Pws << endl;

    auto AddNoise2Obv = [](const Eigen::Vector2d& obv) -> Eigen::Vector2d {
        mt19937 gen(44);
        const double minNoise = 1.0;
        normal_distribution<double> dist(0.0, noiseStd);
        const Eigen::Vector2d noise{dist(gen)+minNoise, dist(gen)+minNoise*0.1};
        const Eigen::Vector2d obv_noisy = obv + noise;
        return obv_noisy;
    };

    auto ProjectPws2CurFrame =
        [&Pws, &depth, &AddNoise2Obv](const Eigen::Matrix3d& Rwc = Eigen::Matrix3d::Identity(),
                       const Eigen::Vector3d& Pwc = {0, 0, 0},
                       const int& time = 0) -> FrameData {
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
            //obvEachFrame.push_back(_obv); // 试试不取整数，不影响逆深度方法2的错误
            _obv[0] = int(_obv[0]);
            _obv[1] = int(_obv[1]);
            obvEachFrame.push_back(AddNoise2Obv(_obv)); // 试试不取整数
            //obvEachFrame.push_back(_obv); // 试试不取整数
            cv::circle(debugImg, cv::Point2i(_obv[0], _obv[1]), 1,
                       cv::Scalar(0, 255, 0), -1);
            //cv::circle(debugImg, cv::Point2i(_obv_noisy[0], _obv_noisy[1]), 1,
            //           {0, 0, 255}, -1);
        }
        //cv::imshow("debug"+to_string(time), debugImg);
        //cv::waitKey();
        return FrameData(Rc_w, Pc_w, depth - Pwc.z(), obvEachFrame, noiseStd, time,
                         debugImg);
    };

    // 初始化世界系的位置
    Eigen::Matrix3d lastRw_c = Eigen::Matrix3d::Identity();
    Eigen::Vector3d lastPw_c = Eigen::Vector3d::Zero();

    // 滑动窗口
    deque<FrameData> historyFrame = {
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

    // 初始化逆深度滤波器
    InverseDepthFilter idepthFilter(historyFrame.front());
    while (1) {
        ++updateTime;
        cout << endl;
        cout << "last Pwc & Rwc in rpy:\n"
             << "lastPwc: " << lastPw_c.transpose() << "\nlastRwc: "
             << RotationMatrixToZYXEulerAngles(lastRw_c).transpose() * kRad2Deg
             << endl;
        // 1. 使用 {0, 0.01, 0}+{0, 0, 0}验证了微小位移可以引起极大的极线方向变化
        // 2. 使用 {0, 0.1, 0} +{0, 0, 30}验证了极线会随着旋转，这是由于图像感光芯片发生了旋转，
        // 虽然真实的极线没有旋转，但是它在成像时，与感光芯片的X轴有了夹角
        // 3. 本质上，极线就是相机C1与地图点Pw连线在相机C2上的投影
        cout << "Please input " << updateTime << "th Pc1_c2: ";
        Eigen::Vector3d Pc1_c2(0, 0, 0);
        cin >> Pc1_c2[0] >> Pc1_c2[1] >> Pc1_c2[2];
        cout << "Get Pc1_c2: " << Pc1_c2.transpose() << endl;
        cout << "Please input " << updateTime << "th Rw_c: ";
        Eigen::Vector3d rpy(0, 0, 0);
        cin >> rpy[0] >> rpy[1] >> rpy[2];
        cout << "Get rpy: " << rpy.transpose() << endl;

        // 产生下一个位姿
        Eigen::Matrix3d Rw_c2 = RPY2Rotation(rpy);
        const Eigen::Vector3d Pw_c2 = lastRw_c * Pc1_c2 + lastPw_c;
        accumulateBaseline += Pc1_c2.norm();
        lastRw_c = Rw_c2;
        lastPw_c = Pw_c2;

        cout << "current accumulateBaseline: " << accumulateBaseline << endl;

        // 注意，当前辅助进近还不需要做的现在的思路这么复杂
        FrameData curFrame = ProjectPws2CurFrame(Rw_c2, Pw_c2, updateTime);

        // 进行极线跟踪，判断特征点1在该图像位置，这个工作量最大
        vector<Eigen::Vector3d> l2s =
            GetAllEpipolarLines(historyFrame, curFrame);
        cout << "l2s.size: " << l2s.size() << endl;

        idepthFilter.UpdateInverseDepth(curFrame, invK);
#define DEBUG_SHOW 1

#if DEBUG_SHOW
        cv::Mat img = curFrame.debugImg.clone();
        for (int i = 0; i < l2s.size(); ++i) {
            const double A = l2s[i].x(), B = l2s[i].y(), C = l2s[i].z();
            cv::Point2f pl(0, -1), pr(img.cols - 20, -1);

            //cout << "l2s[" << i << "]: " << l2s[i].transpose() << endl;
            if (abs(A) < kZeroLine && abs(B) < kZeroLine) {
                continue;
            } else {
                // Ax+By+C=0
                if (abs(B) < kZeroLine) {
                    pl.x = pr.x = -C / A;
                    pl.y = 0;
                    pr.y = img.rows - 1;
                } else if (abs(A) < kZeroLine) {
                    pl.y = pr.y = -C / B;
                } else {
                    pl.y = (-C - pl.x * A) / B;
                    pr.y = (-C - pr.x * A) / B;
                }
                //cout << "pl: " << pl << endl << "pr: " << pr << endl;
                //cv::line(img, {int(pl.x), int(pl.y)}, {int(pr.x), int(pr.y)}, {255, 255, 255}, 1);
                cv::line(img, pl, pr, {255, 255, 255}, 1);
            }
        }
        const vector<Eigen::Vector2d>& obvs = curFrame.obv;
        cv::circle(img, cv::Point2i(obvs[0].x(), obvs[0].y()), 1,
                   cv::Scalar(0, 255, 0), -1);
        cv::circle(img, cv::Point2i(obvs[1].x(), obvs[1].y()), 1, {0, 0, 255},
                   -1);
        //cv::imshow("img", img);
        //cv::waitKey();
#endif

        // 判断是否需要加入该图像到滑窗
        if (Pc1_c2.norm() >= 0.1) {
            historyFrame.push_back(curFrame);
        }
    }

    return 0;
}
