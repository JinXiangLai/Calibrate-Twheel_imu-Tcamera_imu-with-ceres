#pragma once

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
    const ::Eigen::Matrix3d& K) {
    // s * Pn = [Rw|tw] * Pw
    // [Pn]x * [Rw|tw] * Pw = 0 (1)
    // [Pn]x * Rw * Pw = -[Pn]x * tw (2)
    int equationNum = 0;
    // 根据deepseek解释：
    /*
        px = [u, v, 1]
        A = [px]x * K [R|t] * Pw = 0:
        A = [px]x * [PX] = 0，根据叉积公式：
        式(1): v*P[2]*X - 1*P[1]*X
        式(2): 1*P[0]*X - u*P[2]*X
        式(3): u*P[1]*X - v*P[0]*X

        单次观测的等式(3) = -(等式(2)*v + 等式(1)*u)，因此不是线性无关的，
        只有2个有效独立方程
        几何解释为：一条射线只与(x, y)坐标有关，可以确定2个方程
    */
    // 所以 kUsefulConstraintNum只能是2
    constexpr int kUsefulConstraintNum = 2;
    for (const auto& obvs : obvsNorm) {
        equationNum += obvs.size() * kUsefulConstraintNum;
    }

    Eigen::MatrixXd A34, A33;
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

    cout << "A34:\n" << A34 << endl;

    cout << "A33:\n" << A33 << endl;

    Eigen::JacobiSVD<Eigen::MatrixXd> svd(A34, Eigen::ComputeFullV);
    const Eigen::Vector4d bestV =
        svd.matrixV().col(svd.singularValues().size() - 1);
    cout << "singularValues normlized: " << svd.singularValues().transpose()
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
