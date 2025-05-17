#include "utils.h"

#include <iostream>

#include "common.hpp"

using namespace std;

Eigen::Vector3d CrossProduct(const Eigen::Vector3d& a,
                             const Eigen::Vector3d& b) {
    return Eigen::Vector3d{a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2],
                           a[0] * b[1] - a[1] * b[0]};
}

Eigen::Vector3d TransformPw2Pc(const Eigen::Matrix3d& Rc_w,
                               const Eigen::Vector3d& Pc_w,
                               const Eigen::Vector3d& Pw) {
    const Eigen::Matrix<double, 3, 1> Pc = Rc_w * Pw + Pc_w;
    return Pc;
}

Eigen::Vector2d ProjectPc2PixelPlane(const Eigen::Vector3d& K,
                                     const Eigen::Vector3d& Pc) {
    const Eigen::Matrix<double, 3, 1> Pn = Pc / Pc[2];
    const Eigen::Matrix<double, 2, 1> obv = K.block(0, 0, 2, 3) * Pn;
    return obv;
}

Eigen::Vector2d ProjectPw2PixelPlane(const Eigen::Matrix3d& Rc_w,
                                     const Eigen::Vector3d& Pc_w,
                                     const Eigen::Vector3d& K,
                                     const Eigen::Vector3d& Pw) {
    const Eigen::Vector3d& Pc = TransformPw2Pc(Rc_w, Pc_w, Pw);
    const Eigen::Vector2d& obv = ProjectPc2PixelPlane(K, Pc);
    return obv;
}

Eigen::Matrix<double, 2, 3> CalculateObvWrtPwJacobian(
    const Eigen::Matrix3d& Rc_w, const Eigen::Vector3d& Pc_w,
    const Eigen::Matrix3d& K, const Eigen::Vector3d& Pc) {

    const Eigen::Matrix<double, 2, 3> J_r_Pn = K.block(0, 0, 2, 3);
    const double invZ = 1 / Pc[2];
    const double invZ2 = invZ * invZ;
    // clang-format off
    const Eigen::Matrix3d J_Pn_Pc =
        (Eigen::Matrix3d() << invZ, 0, -Pc[0] * invZ2, 
                              0, invZ, -Pc[1] * invZ2,
                              0, 0, 0).finished();
    // clang-format on
    const Eigen::Matrix<double, 2, 3> j = J_r_Pn * J_Pn_Pc * Rc_w;
    return j;
}

std::vector<Eigen::Vector3d> TransformPw2Pc(
    const std::vector<Eigen::Matrix3d>& Rc_ws,
    const std::vector<Eigen::Vector3d>& Pc_ws, const Eigen::Vector3d& Pw) {
    std::vector<Eigen::Vector3d> Pcs(Rc_ws.size());
    for (int i = 0; i < Rc_ws.size(); ++i) {
        Pcs[i] = TransformPw2Pc(Rc_ws[i], Pc_ws[i], Pw);
    }
    return Pcs;
}

void GenerateNextPose(const Eigen::Matrix3d& Rw_c1,
                      const Eigen::Vector3d& Pw_c1,
                      const Eigen::Vector3d& rotAxis,
                      const Eigen::Vector3d& posDirection, const double& rotAng,
                      const double& moveDist, Eigen::Matrix3d& Rw_c2,
                      Eigen::Vector3d& Pw_c2, const bool isDeg) {
    const double rad = isDeg ? rotAng * kDeg2Rad : rotAng;
    Rw_c2 = Eigen::AngleAxisd(rad, rotAxis.normalized()).toRotationMatrix();
    //Rw_c2 = Rw_c1 * Rc1_c2;
    const Eigen::Vector3d Pc1_c2 = moveDist * posDirection.normalized();
    Pw_c2 = Rw_c1 * Pc1_c2 + Pw_c1;
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
    cout << "singularValues normlized: " << svd.singularValues().transpose()
         << endl;
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

Eigen::Vector3d EstimatePwInitialValueOnNormPlane(
    const vector<Eigen::Matrix3d>& Rcw, const vector<Eigen::Vector3d>& Pcw,
    const vector<vector<Eigen::Vector3d>>& obvsNorm, const ::Eigen::Matrix3d& K,
    Eigen::Vector4d& singularValues) {
    // s * Pn = [Rw|tw] * Pw
    // [Pn]x * [Rw|tw] * Pw = 0 (1)
    // [Pn]x * Rw * Pw = -[Pn]x * tw (2) // 预设W不可行
    int equationNum = 0;
    // 根据deepseek解释：
    /*
        px = [u, v, 1]
    推导1：
        A = [px]x * K [R|t] * Pw = 0:
        A = [px]x * [PX] = 0，根据叉积公式：
        式(1): v*P[2]*X - 1*P[1]*X
        式(2): 1*P[0]*X - u*P[2]*X
        式(3): u*P[1]*X - v*P[0]*X

        单次观测的等式(3) = -(等式(2)*v + 等式(1)*u)，因此不是线性无关的，
        只有2个有效独立方程
        几何解释为：一条射线只与(x, y)坐标有关，可以确定2个方程，
        所以 kUsefulConstraintNum只能是2

    推导2：
        Pn = [x, y, 1]
        A = [Pn]x * [R|t] * Pw = [Pn]x * T * Pw = 0，
        系数矩阵关系式为：
        式(1): y*T[2] - 1*T[1]
        式(2): 1*T[0] - x*T[2]
        当仅有绕Z轴的旋转，且沿Z轴运动时：
            |cosθ, -sinθ, 0, tx|
        T = |sinθ,  cosθ, 0, ty|
            |   0,     0, 1, tz|
        其中，tx, ty是常数，记为Cx, Cy
        单个观测的系数矩阵[2x4]为：
              |-sinθ, -cosθ, y, y*tz - ty|
        A_i = |cosθ, -sinθ, -x, tx - x*tz|
        (1): 分析第3列发现，当像素位于图像中心时,(x, y)=(0, 0)，此时Z坐标不可观(其系数均为0)，
            但越远离图像中心，(|x|, |y|)越大，对Z的约束越强
        (2): 分析第1、2列发现，绕Z轴的旋转可以改善对坐标X、Y的约束，
            当无旋转时，cosθ=1, sinθ=0, 则X只能由第2个等式约束，而Y只能由第1个等式约束,
        (3): 尺度W与(x, y)坐标、(tx, ty, tz)有关，当(x, y)靠近像素中心时，(y*tz, x*tz)≈(0, 0),
            若，(tx, ty)越大，对W的约束越明显，但若一直沿Z轴运动，(tx, ty)只提供常数约束，约束能力有限
        (4)：最终，当只有沿Z轴运动时，(X, Y)的约束始终有效，关于Z的约束取决于像素位置，
            关于W的约束也主要取决于像素位置

        当无旋转且仅沿Z轴运动时，A_i退化为：
              |0, -1, y,  y*tz|
        A_i = |1, 0, -x, -x*tz|
        (1) (X, Y)有约束，tz一般取大于0.1m，那么(x, y)在图像的位置是最大的影响因素，
        (2) 我们应该保留越远离图像中心的观测，这样才能提供有效的对Z和W的约束。
        (3) 同样可以发现，当tz越大时，对于W的辨识越明显，意味着越远离图像反而越好估计。
        (4) 由于tz会变换，因此，只能靠第3列来筛选，我们最终应该保留下来第3列值较大，且差异较大的观测组

        结论：
        (1): 在选择有效的约束时，针对每一个观测提供的两个线性无关方程组，可以查找X、Y、Z、W对应的系数最大值，
            然后比较系数量级是否差距过大，因为我们知道，X、Y的系数只与旋转有关，一定是介于[0, 1]之间的数，
            只需比较Z和W对应位置的系数与X和Y量级的差异即可，尽量保留量级差异小的观测.
        (2): 直接按第3、4列中，任意一个观测对应的最大值进行排序取值，同时把初始化的运动距离条件设小，据此
            进行多次初始化
        (3): 优先保留有水平运动，即(tx, ty)变化的帧，其次，再保留像素远离中心的帧以最大限度保证信息有效性
        (4): 设定tz使用的最小值，避免使用tz高度过低的观测
    
    推导3：
        Pn = [x, y, 1]中，[x, y]的取值范围
        x = u/fx - cx/fx
        y = v/fy - cy/fy
        理想情况下，图像分辨率为[WxH]，设：
        cx=fx = W/2, cy=fy = H/2，有：
        x = 2*u/W -1
        y = 2*v/H -1
        u∈[0, W), v∈[0, H)，则归一化坐标取值范围是：
        x∈[-1, 1)
        y∈[-1, 1)
    */
    //
    constexpr int kUsefulConstraintNum = 2;
    for (const auto& obvs : obvsNorm) {
        equationNum += obvs.size() * kUsefulConstraintNum;
    }

    Eigen::MatrixXd A33, A34;
    A34.resize(equationNum, 4);
    // QR分解方法因未使用齐次坐标，则默认假设Z=1，这种解法是不对的
    A33.resize(equationNum, 3);
    Eigen::VectorXd b;
    b.resize(equationNum, 1);
    int rowId = 0;
    for (int i = 0; i < Rcw.size(); ++i) {
        for (int j = 0; j < obvsNorm[i].size(); ++j) {
            const Eigen::Vector3d& obv_i = obvsNorm[i][j];
            const Eigen::Matrix3d skewObv = skew(obv_i);
            // svd求解
            Eigen::Matrix<double, 3, 4> T;
            T.block(0, 0, 3, 3) = Rcw[i];
            T.block(0, 3, 3, 1) = Pcw[i];
            A34.block(rowId, 0, kUsefulConstraintNum, 4) =
                (skewObv * T).block(0, 0, kUsefulConstraintNum, 4);
            // QR分解
            A33.block(rowId, 0, kUsefulConstraintNum, 3) =
                (skewObv * Rcw[i]).block(0, 0, kUsefulConstraintNum, 3);
            b.block(rowId, 0, kUsefulConstraintNum, 1) =
                (-skewObv * Pcw[i]).head(kUsefulConstraintNum);

            rowId += kUsefulConstraintNum;
        }
    }

    // 注意：A34和A33的前3列系数矩阵完全一致
    //cout << "A34:\n" << A34 << endl;
    //cout << "A33:\n" << A33 << endl;

    Eigen::JacobiSVD<Eigen::MatrixXd> svd(A34, Eigen::ComputeFullV);
    singularValues = svd.singularValues();
    const Eigen::Vector4d bestV = svd.matrixV().col(singularValues.size() - 1);
    cout << "singularValues normlized: " << singularValues.transpose() << endl;
    Eigen::Vector3d estNormPw = bestV.head(3) / bestV[3];
    cout << "src estNormPw: " << estNormPw.transpose() << endl;
    if (estNormPw.z() < 0) {
        estNormPw *= -1.0;
    }
    cout << "tar estNormPw: " << estNormPw.transpose() << endl;

    //const Eigen::Vector3d& estPw = A33.colPivHouseholderQr().solve(b);
    const Eigen::Vector3d& estPw = A33.fullPivHouseholderQr().solve(b);

    cout << "QR estPw 不含尺度: " << estPw.transpose() << endl;

    return estNormPw;
    //return estPw;
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
                                const Eigen::Vector3d& Pcw,
                                const Eigen::Matrix3d& K) {
    const Eigen::Vector3d Pc = Rcw * Pw + Pcw;
    const Eigen::Vector3d Pn = Pc / Pc[2];
    const Eigen::Vector2d obv = K.block(0, 0, 2, 3) * Pn;
    return obv;
}

cv::Mat stitchAndDrawMatches(const deque<cv::Mat>& debugImgs,
                             const deque<vector<Eigen::Vector2d>>& obvs) {
    if (debugImgs.empty() || obvs.empty()) {
        return cv::Mat();
    }

    // --- 1. 图像拼接 ---
    // 计算拼接后的总宽度和最大高度
    int totalWidth = 0;
    int maxHeight = 0;
    for (const auto& img : debugImgs) {
        totalWidth += img.cols;
        maxHeight = max(maxHeight, img.rows);
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
    const vector<cv::Scalar> colors = {
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
        vector<cv::Point> points;
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

cv::Mat stitchAndDrawMatches(const deque<DataFrame>& slidingWindow) {
    deque<cv::Mat> debugImgs;
    deque<vector<Eigen::Vector2d>> debugObvs;

    for (const DataFrame& f : slidingWindow) {
        debugObvs.push_back(f.obv);
        debugImgs.push_back(f.debugImg);
    }

    return stitchAndDrawMatches(debugImgs, debugObvs);
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

void CalculateInitialPwDLT(const deque<DataFrame>& slidingWindow,
                           const Eigen::Matrix3d& K, Eigen::Vector3d& initPw,
                           Eigen::Vector4d& singularValues) {
    // 定义中间变量
    vector<Eigen::Matrix3d> Rc_w(slidingWindow.size());
    vector<Eigen::Vector3d> Pc_w(slidingWindow.size());
    vector<double> height2Ground(slidingWindow.size());
    vector<vector<Eigen::Vector2d>> obvs(slidingWindow.size());

    ExtrackPoseAndObvFromSlidingWindow(slidingWindow, Rc_w, Pc_w, obvs);

    // 保存归一化的观测均值与放缩系数
    vector<Eigen::Vector2d> meanObvs(slidingWindow.size());
    vector<vector<double>> scales(slidingWindow.size());
    for (int i = 0; i < slidingWindow.size(); ++i) {
        const DataFrame& frame = slidingWindow[i];
        height2Ground[i] = frame.height2Ground;
        Eigen::Vector2d meanObv = Eigen::Vector2d::Zero();
        for (int j = 0; j < obvs[i].size(); ++j) {
            meanObv += obvs[i][j];
        }
        meanObv /= obvs[i].size();
        scales[i].resize(obvs[i].size());
        for (int j = 0; j < obvs[i].size(); ++j) {
            const double d = (obvs[i][j] - meanObv).norm();
            scales[i][j] = kSqrt2 / d;
        }
    }

#define CALCULATE_INIT_Pw_DLT 1

#if !CALCULATE_INIT_Pw_DLT
    for (int i = 0; i < Pc_w.size(); ++i) {
        cout << "Pw_c[" << i
             << "]: " << (-Rc_w[i].transpose() * Pc_w[i]).transpose() << endl;
        const double& depth = height2Ground[i];
        priorPw += CalculatePriorPwByHeight(Rc_w[i].transpose(),
                                            -Rc_w[i].transpose() * Pc_w[i],
                                            depth, K, obvs[i]);
    }
    priorPw /= Pc_w.size();
    cout << "CalculatePriorPwByHeight: " << priorPw.transpose() << endl;
#else
    // 使用DLT计算初始值
    //const Eigen::Vector3d initPw1 =
    //    EstimatePwInitialValue(Rc_w, Pc_w, obvs, K);
    //const Eigen::Vector3d initPw2 =
    //    EstimatePwInitialValueNormlized(Rc_w, Pc_w, obvs, K);
    const Eigen::Matrix3d invK = K.inverse();
    vector<vector<Eigen::Vector3d>> obvsNorm(slidingWindow.size());
    for (int i = 0; i < obvs.size(); ++i) {
        obvsNorm[i].resize(obvs[i].size());
        for (int j = 0; j < obvs[i].size(); ++j) {
            const Eigen::Vector2d& obvj = obvs[i][j];
            obvsNorm[i][j] = invK * Eigen::Vector3d{obvj[0], obvj[1], 1};
        }
    }

    initPw = EstimatePwInitialValueOnNormPlane(Rc_w, Pc_w, obvsNorm, K,
                                               singularValues);
    cout << "initPw1: " << initPw.transpose() << endl;
    // TODO: 这里可以检测Pw应该小于每个Pwc的值

#endif
}

bool CheckInitialPwValidity(const vector<Eigen::Matrix3d>& Rc_w,
                            const vector<Eigen::Vector3d>& Pc_w,
                            const Eigen::Vector3d& initPw) {

    const vector<Eigen::Vector3d>& Pcs = TransformPw2Pc(Rc_w, Pc_w, initPw);
    for (int i = 0; i < Pcs.size(); ++i) {
        if (Pcs[i].z() < 0) {
            cout << "Pcs[" << i << "].Z==" << Pcs[i].z() << "<0" << endl
                 << "Pw_c: " << (-Rc_w[i].transpose() * Pc_w[i]).transpose()
                 << endl;
            ;
            return false;
        }
    }
    return true;
}

vector<int> GetEraseObservationId(const deque<DataFrame>& slidingWindow) {
    //const int midId = slidingWindow.size() / 2;
    const int midId = 0;  // 保留首帧以保证基线长度持续增长
    std::vector<int> keepIds, removeIds;
    keepIds.push_back(midId);

    int frameId = -1;
    while (frameId < int(slidingWindow.size() - 1)) {
        ++frameId;
        if (frameId == midId) {
            continue;
        }

        // 检验当前id是否值得保留
        bool keep = true;
        for (int i = 0; i < keepIds.size(); ++i) {
            // 检验策略：
            // 1. 因为我们已经有了运动先验，所以可以规避飞机暂停的状态
            // 2. 我们认为，若有水平运动，那么检测像素的偏差应该是较大的，所以这里就不检查水平平移量，一般水平平移量是要优先被考虑的，
            //    但因为其影响W的系数，并且，当飞机距离较远时，其观测像素变化量可能较小，所以还是要独立判断tx, ty变化量
            // 3. 因此，我们同时检查(tx, ty)变化量及像素偏差是否足够大，以确认是否可以保留
            constexpr double minHorDiff = 0.2, minPxDiff2Keep = 3.0;
            const int kr = keepIds[i];
            const double horDiff =
                (slidingWindow[kr].GetPw() - slidingWindow[frameId].GetPw())
                    .head(2)
                    .norm();
            const double pxDiff = (slidingWindow[kr].GetMainObv() -
                                   slidingWindow[frameId].GetMainObv())
                                      .norm();
            // 只有水平运动足够小，且像素变化也不大的才认为是重复观测
            if (horDiff < minHorDiff && pxDiff < minPxDiff2Keep) {
                cout << "remove id=" << frameId << "horDiff=" << horDiff
                     << " & pxDiff=" << pxDiff << endl;
                keep = false;
                break;
            }
        }
        if (keep) {
            keepIds.push_back(frameId);
        } else {
            removeIds.push_back(frameId);
        }
    }

    return removeIds;
}

void ExtrackPoseAndObvFromSlidingWindow(const deque<DataFrame>& slidingWindow,
                                        vector<Eigen::Matrix3d>& Rc_ws,
                                        vector<Eigen::Vector3d>& Pc_ws,
                                        vector<vector<Eigen::Vector2d>>& obvs) {
    Rc_ws.resize(slidingWindow.size());
    Pc_ws.resize(slidingWindow.size());
    obvs.resize(slidingWindow.size());
    for (int i = 0; i < slidingWindow.size(); ++i) {
        Rc_ws[i] = slidingWindow[i].Rc_w;
        Pc_ws[i] = slidingWindow[i].Pc_w;
        obvs[i] = slidingWindow[i].obv;
    }
}

void CullingBadObservationsBeforeInit(deque<DataFrame>& slidingWindow) {
    const vector<int>& mvIds = GetEraseObservationId(slidingWindow);
    for (const int& id : mvIds) {
        slidingWindow[id].timestamp = -1;
        // TODO： 这里删除会导致移位
        cerr << "will remove the id=" << id << " frame" << endl;
        //slidingWindow.erase(slidingWindow.begin() + id);
    }

    // 这里才能安全删除相关量
    auto it = slidingWindow.begin();
    while (it != slidingWindow.end()) {
        if (it->timestamp == -1) {
            cout << "erase it->t: " << it->timestamp << endl;
            it = slidingWindow.erase(it);
        } else {
            ++it;
        }
    }
}

bool CheckLastestObservationUseful(deque<DataFrame>& slidingWindow) {
    const DataFrame& last = slidingWindow.back();
    // 检验当前id是否值得保留
    int keep = true;
    for (int i = 0; i < slidingWindow.size() - 1; ++i) {
        // 检验策略：
        // 1. 因为我们已经有了运动先验，所以可以规避飞机暂停的状态
        // 2. 我们认为，若有水平运动，那么检测像素的偏差应该是较大的，所以这里就不检查水平平移量，一般水平平移量是要优先被考虑的，
        //    因为其影响W的系数，并且，当飞机距离较远时，其观测像素变化量可能较小，所以还是要独立判断tx, ty变化量
        // 3. 因此，我们仅检查像素偏差是否足够大，以确认是否可以保留
        constexpr double minHorDiff = 0.2, minPxDiff2Keep = 3.0;
        const double horDiff =
            (slidingWindow[i].GetPw() - last.GetPw()).head(2).norm();
        const double pxDiff =
            (slidingWindow[i].GetMainObv() - last.GetMainObv()).norm();
        // 只有水平运动足够小，且像素变化也不大的才认为是重复观测
        if (horDiff < minHorDiff && pxDiff < minPxDiff2Keep) {
            cout << "remove the last obv "
                 << "horDiff=" << horDiff << " & pxDiff=" << pxDiff << endl;
            keep = false;
            break;
        }
    }
    if (!keep) {
        slidingWindow.pop_back();
    }

    return keep;
}

// 将旋转矩阵转换为旋转向量，BCH近似时使用
Eigen::Vector3d LogSO3(const Eigen::Matrix3d& R) {
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
    if (costheta > 1 || costheta < -1)
        return w;
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
        return theta * w / s;  // θ/sin(θ)*w
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

Eigen::Matrix3d InverseRightJacobianSO3(const Eigen::Vector3d& v) {
    return InverseRightJacobianSO3(v[0], v[1], v[2]);
}

void PrintReprojectErrorEachFrame(const std::deque<DataFrame>& sw,
                                  const Eigen::Vector3d& Pw,
                                  const Eigen::Matrix3d& K) {
    cout << "each frame obv residual(pixels) in Pw: " << Pw.transpose() << endl;
    for (const DataFrame& f : sw) {
        cout << f.GetObvResidual(Pw, K).norm() << " ";
    }
    cout << endl;

    constexpr double inflatRatio = 100;
    cout << "each frame obv residual(norm plane) in Pw: " << Pw.transpose()
         << endl;
    for (const DataFrame& f : sw) {
        cout << f.GetNormObvResidual(Pw) * inflatRatio << " ";
    }
    cout << endl;
}

Eigen::Matrix3d CalculateHessianMatrix(
    const std::deque<DataFrame>& slidingWindow, const Eigen::Matrix3d& K,
    const Eigen::Vector3d& Pw) {

#define USE_ALL_OBV 1
    Eigen::Matrix3d H = Eigen::Matrix3d::Zero();
    for (int i = 0; i < slidingWindow.size(); ++i) {
        // 观察雅可比我们可以发现：雅可比与量测无关，因此这里只需要计算主量测即可
        const DataFrame& f = slidingWindow[i];
        const Eigen::Vector3d& Pci = f.GetPc(Pw);

        const Eigen::Matrix<double, 2, 3>& Ji =
            CalculateObvWrtPwJacobian(f.Rc_w, f.Pc_w, K, Pci);
        const double wi = 1.0;
#if USE_ALL_OBV
        // 可以看到，如果使用所有量测，那么H将会被放大，相应地
        // cov是其逆就会被缩小
        H += f.obv.size() * wi * Ji.transpose() * Ji;
#else
        H += wi * Ji.transpose() * Ji;
#endif
    }

    return H;
}

bool CalculateCovariance(const Eigen::Matrix3d& H, Eigen::Matrix3d& cov,
                         const double& sigma2) {

    Eigen::JacobiSVD<Eigen::Matrix3d> svd(
        H, Eigen::ComputeFullU | Eigen::ComputeFullV);

    const Eigen::Vector3d& s = svd.singularValues();
    const double invConditionNum = s[2] / s[1];
    if (invConditionNum < kMinReciprocalConditionNumber) {
        cerr << "singularValues: " << s.transpose() << endl
             << "invConditionNum: " << invConditionNum << endl;
        return false;
    }

    const Eigen::Matrix3d& U = svd.matrixU();
    const Eigen::Matrix3d& V = svd.matrixV();
    Eigen::Matrix3d S = Eigen::Matrix3d::Identity();
    S.diagonal() << 1 / s[0], 1 / s[1], 1 / s[2];

    cov = sigma2 * V * S * U.transpose();
    return true;
}

double CalculateChi2Distance(const Eigen::Matrix3d& cov,
                             const Eigen::Vector3d& pos,
                             const Eigen::Vector3d& est) {
    // TODO: 检测cov是否可逆
    const Eigen::Vector3d& diff = est - pos;
    const Eigen::Matrix3d& m = cov.inverse();
    return diff.transpose() * m * diff;
}

Eigen::Vector3d RotationMatrixToZYXEulerAngles(const Eigen::Matrix3d& R) {
    Eigen::Vector3d euler_angles;

    // 计算俯仰角 theta (绕Y轴)
    euler_angles[1] = asin(-R(2, 0));  // theta = asin(-r31)

    // 避免万向节锁（Gimbal Lock）时的数值误差
    const double eps = 1e-6;
    if (std::abs(R(2, 0)) < 1.0 - eps) {
        // 常规情况
        euler_angles[0] = atan2(R(2, 1), R(2, 2));  // phi = atan2(r32, r33)
        euler_angles[2] = atan2(R(1, 0), R(0, 0));  // psi = atan2(r21, r11)
    } else {
        // 万向节锁情况（cos(theta) ≈ 0）
        euler_angles[0] = 0.0;                       // 任意选择 phi
        euler_angles[2] = atan2(-R(0, 1), R(1, 1));  // psi = atan2(-r12, r22)
    }

    return euler_angles;
}
