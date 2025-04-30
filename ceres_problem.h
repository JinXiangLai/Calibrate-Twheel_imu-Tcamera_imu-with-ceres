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
// =======================================================================//
// =======================================================================//

// utils
// 实际实验发现，只要方向对就行，尽管初值很不准，但在所估计的初值方向附近
// 只有最优值这么一个极小值，去验证吧！
// 这里的三角化没有考虑K矩阵的放缩作用，结果极不稳定
Eigen::Vector3d EstimatePwInitialValue(
    const std::vector<Eigen::Matrix3d>& Rcw, const std::vector<Eigen::Vector3d>& Pcw,
    const std::vector<std::vector<Eigen::Vector2d>>& obvs, const ::Eigen::Matrix3d& K);

Eigen::Vector3d EstimatePwInitialValueNormlized(
    const std::vector<Eigen::Matrix3d>& Rcw, const std::vector<Eigen::Vector3d>& Pcw,
    const std::vector<std::vector<Eigen::Vector2d>>& obvs, const ::Eigen::Matrix3d& K);

Eigen::Vector3d EstimatePwInitialValueOnNormPlane(
    const std::vector<Eigen::Matrix3d>& Rcw, const std::vector<Eigen::Vector3d>& Pcw,
    const std::vector<std::vector<Eigen::Vector3d>>& obvsNorm,
    const ::Eigen::Matrix3d& K);
