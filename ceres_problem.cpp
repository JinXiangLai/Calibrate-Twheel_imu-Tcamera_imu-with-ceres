#include <ceres/ceres.h>
#include <ceres/cost_function.h>
#include <ceres/manifold.h>
#include <ceres/sized_cost_function.h>
#include <ceres/types.h>

#include "ceres_problem.h"

using namespace std;

// =======================================================================//
// =======================================================================//
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

ScaledProjectionResidual::ScaledProjectionResidual(
    const Eigen::Matrix3d& _Rc_w, const Eigen::Vector3d& _Pc_w,
    const Eigen::Vector2d& _obv, const Eigen::Matrix3d& _K,
    const Eigen::Vector2d& _meanObv)
    : Rc_w_(_Rc_w), Pc_w_(_Pc_w), obv_(_obv), K(_K), meanObv_(_meanObv) {
    // SizedCostFunction同样继承自CostFunction，
    // 若不使用SizedCostFunction，则需要手动设置残差维度和参数块大小
    set_num_residuals(2);
    // 这里
    mutable_parameter_block_sizes()->push_back(3);

    const double d = (_obv - _meanObv).norm();
    scale_ = kSqrt2 / d;
    sobv_ = Eigen::Vector2d(scale_ * obv_.x() - scale_ * meanObv_.x(),
                            scale_ * obv_.y() - scale_ * meanObv_.y());
}

bool ScaledProjectionResidual::Evaluate(double const* const* parameters,
                                        double* residuals,
                                        double** jacobians) const {
    // 获取优化参数
    const double* p = parameters[0];
    Eigen::Vector3d Pw(p[0], p[1], p[2]);

    // 重投影
    const Eigen::Matrix<double, 3, 1> Pc = Rc_w_ * Pw + Pc_w_;
    const Eigen::Matrix<double, 3, 1> Pn = Pc / Pc[2];
    const Eigen::Matrix<double, 2, 1> obv = K.block(0, 0, 2, 3) * Pn;

    auto GetScaleObv = [this](const Eigen::Vector2d& obv) -> Eigen::Vector2d {
        return Eigen::Vector2d(scale_ * obv.x() - scale_ * meanObv_.x(),
                               scale_ * obv.y() - scale_ * meanObv_.y());
    };
    const Eigen::Vector2d sobv = GetScaleObv(obv);

    // 计算残差
    //residuals[0] = obv(0) - obv_(0);
    //residuals[1] = obv(1) - obv_(1);
    residuals[0] = sobv.x() - sobv_.x();
    residuals[1] = sobv.y() - sobv_.y();

    // 计算雅可比
    if (jacobians) {
        if (jacobians[0]) {
            // Evaluate函数中的雅可比是关于环境维度的，所以维度是[3x4]
            Eigen::Map<Eigen::Matrix<double, 2, 3, Eigen::RowMajor>> j(
                jacobians[0]);
            j.setZero();
            const Eigen::Matrix2d J_r_obv =
                (Eigen::Matrix2d() << scale_, 0, 0, scale_).finished();
            const Eigen::Matrix<double, 2, 3> J_obv_Pn = K.block(0, 0, 2, 3);
            const double invZ = 1 / Pc[2];
            const double invZ2 = invZ * invZ;
            const Eigen::Matrix3d J_Pn_Pc =
                (Eigen::Matrix3d() << invZ, 0, -Pc[0] * invZ2, 0, invZ,
                 -Pc[1] * invZ2, 0, 0, 0)
                    .finished();
            j = J_r_obv * J_obv_Pn * J_Pn_Pc * Rc_w_;
        }
    }

    return true;
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
// =======================================================================//
// =======================================================================//

// utils
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
    const vector<vector<Eigen::Vector3d>>& obvsNorm,
    const ::Eigen::Matrix3d& K, Eigen::Vector4d& singularValues) {
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
    const Eigen::Vector4d bestV =
        svd.matrixV().col(singularValues.size() - 1);
    cout << "singularValues normlized: " << singularValues.transpose()
         << endl;
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
