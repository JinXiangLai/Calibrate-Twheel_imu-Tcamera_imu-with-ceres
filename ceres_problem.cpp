#include "ceres_problem.h"
#include "common.hpp"

using namespace std;

// =======================================================================//
// =======================================================================//
ProjectionResidual::ProjectionResidual(const Eigen::Matrix3d& _Rc_w,
                                       const Eigen::Vector3d& _Pc_w,
                                       const Eigen::Vector2d& _obv,
                                       const Eigen::Matrix3d& _K)
    : Rc_w_(_Rc_w), Pc_w_(_Pc_w), obv_(_obv), K_(_K) {
    // SizedCostFunction同样继承自CostFunction，
    // 若不使用SizedCostFunction，则需要手动设置残差维度和参数块大小
    set_num_residuals(2);
    // 这里
    mutable_parameter_block_sizes()->push_back(3);

#if RESIDUAL_ON_NORMLIZED_PLANE
    obvNorm_ << (obv_[0] - kCx) / kFx,
        (obv_[1] - kCy) / kFy;
#endif
}

bool ProjectionResidual::Evaluate(double const* const* parameters,
                                  double* residuals, double** jacobians) const {
    // 获取优化参数
    const double* p = parameters[0];
    Eigen::Vector3d Pw(p[0], p[1], p[2]);

    // 重投影
    const Eigen::Matrix<double, 3, 1> Pc = Rc_w_ * Pw + Pc_w_;
    const Eigen::Matrix<double, 3, 1> Pn = Pc / Pc[2];

#if RESIDUAL_ON_NORMLIZED_PLANE
    const Eigen::Matrix<double, 2, 1> obv = Pn.head(2);
    // 计算残差
    residuals[0] = obv(0) - obvNorm_(0);
    residuals[1] = obv(1) - obvNorm_(1);
#else
    const Eigen::Matrix<double, 2, 1> obv = K_.block(0, 0, 2, 3) * Pn;
    // 计算残差
    residuals[0] = obv(0) - obv_(0);
    residuals[1] = obv(1) - obv_(1);
#endif

    // 计算雅可比
    if (jacobians) {
        if (jacobians[0]) {
            // Evaluate函数中的雅可比是关于环境维度的，所以维度是[3x4]
            Eigen::Map<Eigen::Matrix<double, 2, 3, Eigen::RowMajor>> j(
                jacobians[0]);
            j.setZero();

            const double invZ = 1 / Pc[2];
            const double invZ2 = invZ * invZ;

#if RESIDUAL_ON_NORMLIZED_PLANE
            // clang-format off
            // J_res_Pn = Eigen::Matrix2d::Identity();
            const Eigen::Matrix<double, 2, 3> J_Pn_Pc =
                (Eigen::Matrix<double, 2, 3>() << invZ, 0, -Pc[0] * invZ2, 
                                                  0, invZ, -Pc[1] * invZ2).finished();
            j = J_Pn_Pc * Rc_w_;
#else
            const Eigen::Matrix3d J_Pn_Pc =
                (Eigen::Matrix3d() << invZ, 0, -Pc[0] * invZ2, 
                                      0, invZ, -Pc[1] * invZ2, 
                                      0, 0, 0).finished();
            const Eigen::Matrix<double, 2, 3> J_r_Pn = K.block(0, 0, 2, 3);
            j = J_r_Pn * J_Pn_Pc * Rc_w_;
            // clang-format on
#endif
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
