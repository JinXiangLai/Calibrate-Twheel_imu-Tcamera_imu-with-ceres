#include "classes.h"

#include <Eigen/Dense>  // 才具有完整的矩阵运算功能

#include "common.hpp"
#include "utils.h"

using namespace std;

FrameData::FrameData(const Eigen::Matrix3d& _Rc_w, const Eigen::Vector3d& _Pc_w,
                     const double& height,
                     const std::vector<Eigen::Vector2d>& _obv, const double& _t,
                     const cv::Mat& img)
    : timestamp(_t),
      Rc_w(_Rc_w),
      Pc_w(_Pc_w),
      height2Ground(height),
      obv(_obv),
      debugImg(img) {
    obvNorm << (obv[0].x() - kCx) / kFx, (obv[0].y() - kCy) / kFy;
    return;
}

void PriorEstimateData::UpdateMean(const Eigen::Vector3d& p) {
    const Eigen::Vector3d sum = updateCount_ * Pw_ + p;
    ++updateCount_;
    Pw_ = sum / updateCount_;
    historyEstPw_.emplace_back(p);
}

void PriorEstimateData::UpdateMessageMatrix(const Eigen::Matrix3d& cov) {
    Eigen::Matrix3d C = cov;
    const double ratio = C.diagonal().norm() / (3 * 0.15);
    if (ratio < 1) {
        C.diagonal() /= ratio;
    }
    //C.diagonal() += Eigen::Vector3d::Ones() * 1e-15;
    const Eigen::Matrix3d& h = C.inverse();
    H_ += h;
    historyEstCov_.emplace_back(cov);
}

void PriorEstimateData::UpdateMeanAndH(const Eigen::Vector3d& p,
                                       const Eigen::Matrix3d& cov) {
    UpdateMean(p);
    UpdateMessageMatrix(cov);
}

Eigen::Vector2d FrameData::GetObvResidual(const Eigen::Vector3d& Pw,
                                          const Eigen::Matrix3d& K) const {
    const Eigen::Vector3d& Pc = GetPc(Pw);
    const Eigen::Vector3d& Pn = Pc / Pc.z();
    const Eigen::Vector2d& estObv = K.block(0, 0, 2, 3) * Pn;
    return (estObv - obv[0]);
}

double FrameData::GetNormObvResidual(const Eigen::Vector3d& Pw) const {
    const Eigen::Vector3d& Pc = Rc_w * Pw + Pc_w;
    const Eigen::Vector3d& Pn = Pc / Pc.z();
    return (Pn.head(2) - obvNorm).norm();
}

InverseDepthFilter::InverseDepthFilter(const FrameData& curF)
    : host_(curF), last_(curF) {
    idepth_ = 1.0 / 10.;
    const double depthMin = 1.0, depthMax = 100.0;
    cov_ = std::pow((1.0 / depthMin - 1.0 / depthMax) / std::sqrt(12), 2);
}

InverseDepthFilter::InverseDepthFilter(const double& idepth,
                                       const double& depthMin,
                                       const double& depthMax,
                                       const FrameData& curF)
    : host_(curF), last_(curF) {
    idepth_ = idepth;
    cov_ = std::pow((1.0 / depthMin - 1.0 / depthMax) / std::sqrt(12), 2);
    initialized_ = true;
}

// 鲁棒性更新
bool InverseDepthFilter::UpdateWithRobustCheck(const double& idepthObs,
                                               const double& obsNoisepixel,
                                               const double& baseline) {
    const double cov_obs = std::pow(obsNoisepixel / baseline, 2);
    const double residual = idepthObs - idepth_;
    const double residualStd = std::sqrt(cov_ + cov_obs);

    if (residual * residual / (cov_ + cov_obs) >
        3.84) {       // 3.84 自由度为1的卡方分布
        cov_ *= 1.5;  // 异常值处理
        return false;
    }

    const double denominator = cov_ + cov_obs;
    idepth_ = (idepth_ * cov_ + idepthObs * cov_obs) / denominator;
    cov_ = (cov_ * cov_obs) / denominator;
    return true;
}

// 坐标系变换（带协方差传播）
bool InverseDepthFilter::TransformHost(const FrameData& curF,
                                       const Eigen::Matrix3d& invK) {
    const Eigen::Vector3d Pc1 =
        invK * Eigen::Vector3d(host_.obv[0].x(), host_.obv[0].y(), 1.0);
    const Eigen::Matrix3d Rc2_c1 = curF.Rc_w * host_.Rc_w.transpose();
    const Eigen::Vector3d Pc2_c1 = curF.Rc_w * (host_.GetPw() - curF.GetPw());
    const Eigen::Vector3d Pc2 = Rc2_c1 * Pc1 + Pc2_c1 * idepth_;

    if (Pc2.z() <= 0) {
        cov_ *= 2.0;  // 失效处理
        return false;
    }

    // 1.0/ρ2 * Pn2 = R21 * 1.0/ρ1 * Pn1 + P21
    // 1.0/ρ2 * (Pn2.T * Pn2) = 1.0/ρ1 * (Pn2.T * R21 * Pn1) + (Pn2.T * P21)
    // 1.0/ρ2 * A = 1.0/ρ1 * B + C
    // 1.0/ρ2 = 1.0/ρ1 * B/A + C/A
    // ρ2 = A/(1.0/ρ1 * B + C) = A/t

    // dρ2/dρ1 = -A/t^2 * -B/ρ1^2 = AB/(t*ρ1)^2

    // 正确推导核心：直接在第一步取z分量推导即可
    // 1.0/ρ2 = 1.0/ρ1 * (R21 * Pn1)z + (P21)z
    // ρ2 = 1.0 / (1.0/ρ1 * (R21 * Pn1)z + (P21)z) = 1.0 / (1.0/ρ1 * A + B)

    // dρ2/dρ1 = -1.0/(1.0/ρ1 * A + B)^2 * -A/ρ1^2 = A/(1.0/ρ1 * A * ρ1  + B * ρ1)^2 = A/(A+B*ρ1)^2
    const double J = Pc2_c1.z() / (Pc2.z() * Pc2.z());
    idepth_ = 1.0 / Pc2.z();
    cov_ = J * cov_ * J;
    host_ = curF;
    return true;
}

bool InverseDepthFilter::Update(const FrameData& curF) {}
