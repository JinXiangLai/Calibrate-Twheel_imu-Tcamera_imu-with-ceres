#pragma once

#include <ceres/ceres.h>
#include <ceres/cost_function.h>
#include <ceres/manifold.h>
#include <ceres/sized_cost_function.h>
#include <ceres/types.h>

#include "common.hpp"

// =======================================================================//
// =======================================================================//

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

class ScaledProjectionResidual : public ceres::CostFunction {
   public:
    ScaledProjectionResidual(const Eigen::Matrix3d& _Rc_w,
                             const Eigen::Vector3d& _Pc_w,
                             const Eigen::Vector2d& _obv,
                             const Eigen::Matrix3d& _K,
                             const Eigen::Vector2d& _meanObv);
    virtual bool Evaluate(double const* const* parameters, double* residuals,
                          double** jacobians) const override;

   private:
    Eigen::Matrix3d Rc_w_;
    Eigen::Vector3d Pc_w_;
    Eigen::Vector2d obv_;
    Eigen::Matrix3d K;

    Eigen::Vector2d meanObv_;
    Eigen::Vector2d sobv_;
    double scale_ = 0;
};

// 自定义四元数的 LocalParameterization
class QuaternionParameterization : public ceres::Manifold {
   public:
    bool Plus(const double* x, const double* delta,
              double* x_plus_delta) const override {
        // 将 delta 转换为四元数
        Eigen::Matrix<double, 3, 1> delta_q_vec(delta[0], delta[1], delta[2]);
        double theta = delta_q_vec.norm();
        Eigen::Quaterniond delta_q;
        if (theta > 0.0) {
            delta_q = Eigen::Quaterniond(
                Eigen::AngleAxisd(theta, delta_q_vec / theta));
        } else {
            delta_q = Eigen::Quaterniond(1.0, 0.0, 0.0, 0.0);
        }

        // 将 x 转换为四元数
        Eigen::Quaterniond x_q(x[0], x[1], x[2], x[3]);

        // 更新四元数
        Eigen::Quaterniond x_plus_delta_q = x_q * delta_q;
        x_plus_delta_q.normalize();

        // 将结果写回
        x_plus_delta[0] = x_plus_delta_q.w();
        x_plus_delta[1] = x_plus_delta_q.x();
        x_plus_delta[2] = x_plus_delta_q.y();
        x_plus_delta[3] = x_plus_delta_q.z();

        return true;
    }

    virtual bool ComputeJacobian(const double* x, double* jacobian) const {
        return false;
    }

    virtual bool RightMultiplyByPlusJacobian(
        const double* x, const int num_rows, const double* ambient_matrix,
        double* tangent_matrix) const override {
        return false;
    }

    virtual bool Minus(const double* x, const double* y,
                       double* y_minus_x) const override {
        return false;
    }

    virtual bool PlusJacobian(const double* x,
                              double* jacobian) const override {
        // 计算的是Plus函数对delta增量的雅可比矩阵

#if USE_AUTO_DIFF
        Eigen::Map<Eigen::Matrix<double, 3, 4, Eigen::RowMajor>> j(jacobian);
        j.setZero();
        // j(1, 0) = 1.0;
        // j(2, 1) = 1.0;
        // j(3, 2) = 1.0;
        // 以下参考deepseek的推导，使用下面的公式，收敛明显加快，
        // 相同条件下，下面的雅可比只需迭代5次，
        // 上面的雅可比需要迭代48次
        // clang-format off
        j << -0.5 * x[1], -0.5 * x[2], -0.5 * x[3], 0.5 * x[0], 
            -0.5 * x[3], 0.5 * x[2], 0.5 * x[3], 0.5 * x[0], 
            -0.5 * x[1], -0.5 * x[2], 0.5 * x[1], 0.5 * x[0];
        // clang-format on
#else
        // 因为我在Evaluate函数中已经直接给出∂e/∂δ，所以这里只需要保证乘上∂e/∂δ值不变即可
        // 但此时只能使用ceres::LINE_SEARCH，而不能使用ceres::TRUST_REGION
        Eigen::Map<Eigen::Matrix<double, 4, 3, Eigen::RowMajor>> j(jacobian);
        j.setZero();
        j.block(0, 0, 3, 3).setIdentity();
#endif
        return true;
    }

    virtual bool MinusJacobian(const double* x,
                               double* jacobian) const override {
        return false;
    }

    int AmbientSize() const override { return 4; }  // 四元数的环境维度
    int TangentSize() const override { return 3; }  // 四元数的切空间维度
};

// =======================================================================//
// =======================================================================//

// utils
// 实际实验发现，只要方向对就行，尽管初值很不准，但在所估计的初值方向附近
// 只有最优值这么一个极小值，去验证吧！
// 这里的三角化没有考虑K矩阵的放缩作用，结果极不稳定
Eigen::Vector3d EstimatePwInitialValue(
    const std::vector<Eigen::Matrix3d>& Rcw,
    const std::vector<Eigen::Vector3d>& Pcw,
    const std::vector<std::vector<Eigen::Vector2d>>& obvs,
    const ::Eigen::Matrix3d& K);

Eigen::Vector3d EstimatePwInitialValueNormlized(
    const std::vector<Eigen::Matrix3d>& Rcw,
    const std::vector<Eigen::Vector3d>& Pcw,
    const std::vector<std::vector<Eigen::Vector2d>>& obvs,
    const ::Eigen::Matrix3d& K);

Eigen::Vector3d EstimatePwInitialValueOnNormPlane(
    const std::vector<Eigen::Matrix3d>& Rcw,
    const std::vector<Eigen::Vector3d>& Pcw,
    const std::vector<std::vector<Eigen::Vector3d>>& obvsNorm,
    const ::Eigen::Matrix3d& K);
