#include "classes.h"

#include <Eigen/Dense>  // 才具有完整的矩阵运算功能

#include "common.hpp"
#include "utils.h"

using namespace std;

DataFrame::DataFrame(const Eigen::Matrix3d& _Rc_w, const Eigen::Vector3d& _Pc_w,
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

double DataFrame::GetObvResidual(const Eigen::Vector3d& Pw,
                                 const Eigen::Matrix3d& K) const {
    const Eigen::Vector3d& Pc = Rc_w * Pw + Pc_w;
    const Eigen::Vector3d& Pn = Pc / Pc.z();
    const Eigen::Vector2d& estObv = K.block(0, 0, 2, 3) * Pn;
    return (estObv - obv[0]).norm();
}

double DataFrame::GetNormObvResidual(const Eigen::Vector3d& Pw) const {
    const Eigen::Vector3d& Pc = Rc_w * Pw + Pc_w;
    const Eigen::Vector3d& Pn = Pc / Pc.z();
    return (Pn.head(2) - obvNorm).norm();
}
