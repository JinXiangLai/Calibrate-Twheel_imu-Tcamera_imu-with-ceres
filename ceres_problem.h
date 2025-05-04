#pragma once

#include <ceres/ceres.h>  // 包含了ceres全部的头文件
//#include <ceres/cost_function.h>
//#include <ceres/manifold.h>
//#include <ceres/sized_cost_function.h>
//#include <ceres/types.h>

#include <Eigen/Dense>

#include "classes.h"
#include "utils.h"

#define USE_AUTO_DIFF 0
#define ROT_RES_DIM 3

#define RESIDUAL_ON_NORMLIZED_PLANE 0

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
    Eigen::Matrix3d K_;

#if RESIDUAL_ON_NORMLIZED_PLANE
    Eigen::Vector2d obvNorm_;
#endif
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

#if USE_AUTO_DIFF
// 定义误差函数，自动求导条件下，不需定义Evaluate函数，但必须提供operator()函数以计算残差
struct RelativePoseResidual {
    RelativePoseResidual(const Pose& Tb1b2, const Pose& Tc1c2)
        : Tb1b2_(Tb1b2), Tc1c2_(Tc1c2) {}

    // 使用自动求导时才必须提供该函数
    template <typename T>
    bool operator()(const T* const qcb, const T* const pcb, T* residual) const {
        // 将四元数转换为 Eigen::Quaternion
        const Eigen::Quaterniond& q_b1b2 = Tb1b2_.q_wb_;
        const Eigen::Quaterniond& q_c1c2 = Tc1c2_.q_wb_;
        const Eigen::Vector3d& p_b1b2 = Tb1b2_.p_wb_;
        const Eigen::Vector3d& p_c1c2 = Tc1c2_.p_wb_;

        Eigen::Quaternion<T> q_cb(qcb[0], qcb[1], qcb[2], qcb[3]);
        Eigen::Matrix<T, 3, 1> p_cb(pcb[0], pcb[1], pcb[2]);

        // 计算残差 Tcb*Tb1b2 = Tc1c2*Tcb ==>
        // 旋转残差，因为ep也耦合了旋转误差，我们是否可以先使用最小二乘计算一个校准的Rcb，然后再在ep中联合优化呢？
        // 但是最小二乘的本质也是右边的四元数是0向量了
        // er = q_cb*(q_b1b2[r]-q_c1c2[l])
#if ROT_RES_DIM == 4
        const Eigen::Quaternion<T> q1 =
            q_cb * q_b1b2.cast<T>();  // 没有重载减法
        const Eigen::Quaternion<T> q2 = q_c1c2.cast<T>() * q_cb;
        Eigen::Matrix<T, 4, 1> er;
        er << q1.w() - q2.w(), q1.x() - q2.x(), q1.y() - q2.y(),
            q1.z() - q2.z();
#elif ROT_RES_DIM == 3
        Eigen::Quaternion<T> eq = q_cb * Tb1b2_.q_wb_.cast<T>() *
                                  q_cb.inverse() *
                                  Tc1c2_.q_wb_.cast<T>().inverse();
        // const Eigen::Matrix<T, 3, 1> er=LogSO3(eq.template
        // cast<double>().toRotationMatrix()).template cast<T>();
        Eigen::Quaterniond eqd = Eigen::Quaterniond::Identity();
        // eqd.w() = ceres::Jet<double, 7>(q_cb.w()).a;
        // eqd.x() = ceres::Jet<double, 7>(q_cb.x()).a;
        // eqd.y() = ceres::Jet<double, 7>(q_cb.y()).a;
        // eqd.z() = ceres::Jet<double, 7>(q_cb.z()).a;
        eqd.w() = ceres::Jet<double, 7>(eq.w()).a;
        eqd.x() = ceres::Jet<double, 7>(eq.x()).a;
        eqd.y() = ceres::Jet<double, 7>(eq.y()).a;
        eqd.z() = ceres::Jet<double, 7>(eq.z()).a;
        // cout << "eqd: " << eqd.w() << " " << eqd.x() << " " << eqd.y() << " "
        // << eqd.z() << endl;
        Eigen::Vector3d erd = LogSO3(eqd.toRotationMatrix());
        const Eigen::Matrix<T, 3, 1> er = erd.cast<T>();
#endif

        // 平移残差
        const Eigen::Matrix<T, 3, 1> ep =
            q_cb.toRotationMatrix() * p_b1b2.cast<T>() +
            (Eigen::Matrix<T, 3, 3>::Identity() -
             q_c1c2.cast<T>().toRotationMatrix()) *
                p_cb -
            p_c1c2.cast<T>();

        // 计算残差
        // 不能直接赋值，因为residual是double类型的指针，而不是Eigen::Matrix类型的指针
        // Eigen::Map<Eigen::Matrix<T, 7, 1, Eigen::RowMajor>> r(residual);
        // r << er[0], er[1], er[2], er[3], ep[0], ep[1], ep[2];
        for (int i = 0; i < int(ROT_RES_DIM); ++i) {
            residual[i] = T(100) * er[i];
        }
        for (int i = 0; i < 3; ++i) {
            residual[i + int(ROT_RES_DIM)] = ep[i];
        }

        return true;
    }

   private:
    const Pose Tb1b2_;  // IMU轨迹
    const Pose Tc1c2_;  // 相机轨迹
};

#else

// 第2个参数对应Manifold的环境维度还是切空间维度呢？
class RelativePoseResidual : public ceres::CostFunction {
   public:
    RelativePoseResidual(const Pose& Tb1b2, const Pose& Tc1c2)
        : Tb1b2_(Tb1b2), Tc1c2_(Tc1c2) {
        // SizedCostFunction同样继承自CostFunction，
        // 若不使用SizedCostFunction，则需要手动设置残差维度和参数块大小
        set_num_residuals(6);                           // [er, ep]
        mutable_parameter_block_sizes()->push_back(4);  // q_cb
        mutable_parameter_block_sizes()->push_back(3);  // p_cb
    }

    // 使用解析雅可比需要提供Evaluate函数，此时不需要提供operator()函数
    // 所以这里是自动求导和解析雅可比的区别
    // 注意：**parameters这里是二维指针
    virtual bool Evaluate(double const* const* parameters, double* residuals,
                          double** jacobians) const override {
        // Ensure parameters are not null
        if (!parameters[0] || !parameters[1]) {
            std::cerr << "Error: Null parameter pointer." << std::endl;
            return false;
        }

        // Ensure residuals are not null
        if (!residuals) {
            std::cerr << "Error: Null residuals pointer." << std::endl;
            return false;
        }
        // 将四元数转换为 Eigen::Quaternion
        const double* paraQ = parameters[0];
        const double* paramP = parameters[1];
        Eigen::Quaternion<double> q(paraQ[0], paraQ[1], paraQ[2], paraQ[3]);
        Eigen::Vector3d p(paramP[0], paramP[1], paramP[2]);
        // cout << "ceres q: " << q.w() << " " << q.x() << " " << q.y() << " "
        // << q.z() << endl; cout << "ceres p: " << p.transpose() << endl;
        const Eigen::Quaterniond eq =
            q * Tb1b2_.q_wb_ * q.inverse() * Tc1c2_.q_wb_.inverse();
        Eigen::Vector3d er = LogSO3(eq.toRotationMatrix());

        const Eigen::Vector3d ep =
            q * Tb1b2_.p_wb_ +
            (Eigen::Matrix3d::Identity() - Tc1c2_.q_wb_.toRotationMatrix()) *
                p -
            Tc1c2_.p_wb_;

        // 计算残差，修改权重之后，问题可以解决，所以问题的症结在哪里
        // 问题出在推导雅可比，提取公因式时，搞错了一个正负号
        residuals[0] = 100 * er[0];
        residuals[1] = 100 * er[1];
        residuals[2] = 100 * er[2];  // 似乎是该维度的雅可比不准？？
        residuals[3] = 1 * ep[0];
        residuals[4] = 1 * ep[1];
        residuals[5] = 1 * ep[2];

        // cout << "residuals: " << residuals[0] << " " << residuals[1] << " "
        // << residuals[2] << " " << residuals[3] << " " << residuals[4] << " "
        // << residuals[5] << endl;

        // 手动提供解析雅可比矩阵，注意：这里需要忽略旋转的二阶小量
        // 对于Manifold，这里提供的是关于环境维度的雅可比矩阵
        // 所以不能直接使用李代数的结果来赋值，这里提供∂e/∂q，而不是∂e/∂δ
        // 由于四元数可以直接转为旋转矩阵，因此e=q*v=R_q*v，这样就可以得到e关于[qw,
        // qx, qy, qz]的雅可比矩阵了 但这种做法很明显比较复杂，不是我的首选

        // 因为有些时候如在TRUST_REGION策略下，只需要计算residuals
        if (jacobians) {
            if (jacobians[0]) {
                // Evaluate函数中的雅可比是关于环境维度的，所以维度是[3x4]
                Eigen::Map<Eigen::Matrix<double, 6, 4, Eigen::RowMajor>> j(
                    jacobians[0]);
                // 右乘扰动更新雅可比
                // 这里给出了∂e/∂δ的李代数雅可比，因此要注意PlusJacobian中要求实现∂q/∂δ不再是真实值，
                // 而是要保证该处雅可比j右乘上它，矩阵值可以保持不变。
                // 此外需要注意，不能使用ceres::TRUST_REGION，只能使用ceres::LINE_SEARCH，
                // 因为PlusJacobian中的雅可比矩阵不是真实的，ceres2不能用其计算信任域
                j.setZero();
                j.block(0, 0, 3, 3) =
                    InverseRightJacobianSO3(er) *
                    (Tc1c2_.q_wb_ * q).toRotationMatrix() *
                    (Tb1b2_.q_wb_.inverse().toRotationMatrix() -
                     Eigen::Matrix3d::Identity());

                // 只给出这个雅可比时，优化结果也近似正确，所以问题是上述关于旋转的雅可比存在问题？
                j.block(3, 0, 3, 3) =
                    -q.toRotationMatrix() *
                    skew(
                        Tb1b2_
                            .p_wb_);  // 关闭以只优化旋转，测试所推导旋转雅可比的正确性
            }

            if (jacobians[1]) {
                Eigen::Map<Eigen::Matrix<double, 6, 3, Eigen::RowMajor>> j(
                    jacobians[1]);
                j.setZero();
                // j.block(3, 3, 3, 3) =
                // (Eigen::Matrix3d::Identity()-Tc1c2_.q_wb_.toRotationMatrix());
                // 注意，这里jacobians[1]仅是对于第1个状态量的雅可比，最多仅有3列，而不是6列，所以上述写法出错！！！
                j.block(3, 0, 3, 3) =
                    (Eigen::Matrix3d::Identity() -
                     Tc1c2_.q_wb_
                         .toRotationMatrix());  // 关闭以只优化旋转，测试所推导旋转雅可比的正确性
            }
        }

        return true;
    }

   private:
    const Pose Tb1b2_;  // IMU轨迹
    const Pose Tc1c2_;  // 相机轨迹
};
#endif

#if USE_AUTO_DIFF
// 定义误差函数，自动求导条件下，不需定义Evaluate函数，但必须提供operator()函数以计算残差
struct RotationResidual {
    RotationResidual(const Eigen::Vector3d& v_bi, const Eigen::Vector3d& v_li)
        : v_bi_(v_bi), v_li_(v_li) {}

    // 使用自动求导时才必须提供该函数
    template <typename T>
    bool operator()(const T* const q, T* residual) const {
        // 将四元数转换为 Eigen::Quaternion
        Eigen::Quaternion<T> q_rot(q[0], q[1], q[2], q[3]);

        // 将轮速计速度转换为 Eigen::Vector3
        Eigen::Matrix<T, 3, 1> v_li_T = v_li_.cast<T>();

        // 计算旋转后的速度
        Eigen::Matrix<T, 3, 1> predicted_v_bi = q_rot * v_li_T;

        // 计算残差
        residual[0] = predicted_v_bi(0) - T(v_bi_(0));
        residual[1] = predicted_v_bi(1) - T(v_bi_(1));
        residual[2] = predicted_v_bi(2) - T(v_bi_(2));

        return true;
    }

   private:
    const Eigen::Vector3d v_bi_;  // IMU 速度
    const Eigen::Vector3d v_li_;  // 轮速计速度
};

#else

// 第2个参数对应Manifold的环境维度还是切空间维度呢？
class RotationResidual : public ceres::CostFunction {
   public:
    RotationResidual(const Eigen::Vector3d& v_bi, const Eigen::Vector3d& v_li)
        : v_bi_(v_bi), v_li_(v_li) {
        // SizedCostFunction同样继承自CostFunction，
        // 若不使用SizedCostFunction，则需要手动设置残差维度和参数块大小
        set_num_residuals(3);
        // 这里
        mutable_parameter_block_sizes()->push_back(4);
    }

    // 使用解析雅可比需要提供Evaluate函数，此时不需要提供operator()函数
    // 所以这里是自动求导和解析雅可比的区别
    virtual bool Evaluate(double const* const* parameters, double* residuals,
                          double** jacobians) const override {
        // 将四元数转换为 Eigen::Quaternion
        const double* q = parameters[0];
        Eigen::Quaternion<double> q_rot(q[0], q[1], q[2], q[3]);

        // 将轮速计速度转换为 Eigen::Vector3
        Eigen::Matrix<double, 3, 1> v_li_T = v_li_.cast<double>();

        // 计算旋转后的速度
        Eigen::Matrix<double, 3, 1> predicted_v_bi = q_rot * v_li_T;

        // 计算残差
        residuals[0] = predicted_v_bi(0) - v_bi_(0);
        residuals[1] = predicted_v_bi(1) - v_bi_(1);
        residuals[2] = predicted_v_bi(2) - v_bi_(2);

        // 手动提供解析雅可比矩阵
        // 对于Manifold，这里提供的是关于环境维度的雅可比矩阵
        // 所以不能直接使用李代数的结果来赋值，这里提供∂e/∂q，而不是∂e/∂δ
        // 由于四元数可以直接转为旋转矩阵，因此e=q*v=R_q*v，这样就可以得到e关于[qw,
        // qx, qy, qz]的雅可比矩阵了 但这种做法很明显比较复杂，不是我的首选

        // 这里需要首先判断jacobians是否为0x0，因为在TRUST_REGION策略下，我们调用Evaluate函数的目的可能只是需要计算residuals
        if (jacobians) {
            if (jacobians[0]) {
                // Evaluate函数中的雅可比是关于环境维度的，所以维度是[3x4]
                Eigen::Map<Eigen::Matrix<double, 3, 4, Eigen::RowMajor>> j(
                    jacobians[0]);
                // 右乘扰动更新雅可比
                // 这里给出了∂e/∂δ的李代数雅可比，因此要注意PlusJacobian中要求实现∂q/∂δ不再是真实值，
                // 而是要保证该处雅可比j右乘上它，矩阵值可以保持不变。
                // 此外需要注意，不能使用ceres::TRUST_REGION，只能使用ceres::LINE_SEARCH，
                // 因为PlusJacobian中的雅可比矩阵不是真实的，ceres2不能用其计算信任域
                j.setZero();
                j.block(0, 0, 3, 3) = -q_rot.toRotationMatrix() * skew(v_li_T);

                // 还是core dump
                // Eigen::Matrix<double, 3, 4> J_w_q;
                // J_w_q << -0.5 * q[1], -0.5 * q[2], -0.5 * q[3], 0.5 * q[0],
                //             0.5 * q[0], -0.5 * q[3],  0.5 * q[2], 0.5 * q[1],
                //             0.5 * q[3],  0.5 * q[0], -0.5 * q[1], 0.5 * q[2];
                // j.block(0, 0, 3, 4) = j.block(0, 0, 3, 3)*J_w_q;
            }
        }

        return true;
    }

   private:
    const Eigen::Vector3d v_bi_;  // IMU 速度
    const Eigen::Vector3d v_li_;  // 轮速计速度
};
#endif
