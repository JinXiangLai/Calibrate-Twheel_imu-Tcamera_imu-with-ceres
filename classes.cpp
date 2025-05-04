#include "classes.h"

#include <Eigen/Dense>  // 才具有完整的矩阵运算功能

#include "utils.h"

using namespace std;

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
                                 const Eigen::Matrix3d& K) {
    const Eigen::Vector3d& Pc = Rc_w * Pw + Pc_w;
    const Eigen::Vector2d& estObv = K.block(0, 0, 2, 3) * (Pc.head(2) / Pc.z());
    return (estObv - obv[0]).norm();
}
