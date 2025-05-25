#include "classes.h"

#include <Eigen/Dense>  // 才具有完整的矩阵运算功能

#include "common.hpp"
#include "utils.h"

using namespace std;

constexpr double kMinDepth = 10.0;

constexpr double kInitializeRandomDepth = 1.0;

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

bool InverseDepthFilter::RobustChi2Check(const double& idepthObv, const double& covObv) {
    const double residual = idepth_ - idepthObv;
    const double fuseCov = cov_ + covObv;
    const double r2 = residual * residual;
    const double chi2 = r2 / fuseCov;

    constexpr double kChi2Th = 3.84;
    if(chi2 > kChi2Th) {
        cout << "Fail chi2 check for residual=" << residual << " r^2=" << r2 << endl;
        return false;
    }

    cout << "Pass chi2 check for residual=" << residual << " r^2=" << r2 << endl;
    return true;
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

bool InverseDepthFilter::UpdateInverseDepth(const FrameData& curF,
                                            const Eigen::Matrix3d& invK) {
    // 1.0/ρ1 * Pn1 = R12 * 1.0/ρ2 * Pn2 + P12
    // [Pn1 - R12*Pn2]_[3x2] * [1.0/ρ1, 1.0/ρ2] = P12 (1)

    // [Pn2]x * Pn1 * 1.0/ρ1 = [Pn2]x * P12 
    // 1.0/ρ1 = ([Pn2]x * Pn1).T * [Pn2]x * P12 / (([Pn2]x * Pn1).T * [Pn2]x * Pn1) (2)
    // 注意：这里不能再化简，再乘以Pn1.T就变成 0=0了
    const Eigen::Vector3d Pn1 =
        invK * Eigen::Vector3d(host_.obv[0].x(), host_.obv[0].y(), 1.0);
    const Eigen::Vector3d Pn2 =
        invK * Eigen::Vector3d(curF.obv[0].x(), curF.obv[0].y(), 1.0);

    const Eigen::Matrix3d R12 = host_.Rc_w * curF.Rc_w.transpose();
    const Eigen::Vector3d P12 = host_.Rc_w * (curF.GetPw() - host_.GetPw());
    // 第一种解法
    Eigen::Matrix<double, 3, 2> A0 = Eigen::Matrix<double, 3, 2>::Zero();
    A0.col(0) = Pn1;
    A0.col(1) = -R12 * Pn2;
    const Eigen::Vector2d res0 = A0.fullPivHouseholderQr().solve(P12);  // A.jacobiSvd().solve(b);

    // 使用B/A计算出来的方法2比上述最小二乘误差大得多的原因不在于输入观测是否有取整数
    // 理论偏差分析：
    // 公式 (1)：直接解得，无偏差
    // 公式 (2)：若P12含平行于Pn2的分量，则会引入系统性，则叉积会丢失该分量，导致解偏离真实值。
    // 检查公式(2)分母阈值：若<ϵ，判定为退化场景
    const Eigen::Vector3d temp = skew(Pn2) * Pn1;
    const double A = temp.dot(temp);
    const double B = temp.transpose() * (skew(Pn2) * P12);
    cout << "Pn1: " << Pn1.head(2).transpose()
         << " Pn2: " << Pn2.head(2).transpose() << " P12: " << P12.transpose()
         << " A:" << A << " B:" << B << endl;

    Eigen::Vector2d res(B / A, 0);
    cout << "res0[0]: " << res0[0] << endl;
    cout << "res[0]: " << res[0] << endl;

    // 最终还是要使用最小二乘解法
    res = res0;

    if(kInitializeRandomDepth > 0 && !initialized_) {
        initialized_ = true;
        return true;
    }

    if (res[0] < kMinDepth && kInitializeRandomDepth <= 0) {
        return false;
    }
    if (!initialized_) {
        idepth_ = 1 / res[0];
        s_ = res[0];
        initialized_ = true;
        return true;
    }

    cout << "src depth_: " << 1.0 / idepth_ << " estimate depth: " << res[0]
         << endl;
    // 更新协方差

    // 计算N个像素偏差引起的深度不确定度
    // 原理：根据极线约束，认为在极线上存在N个像素偏差时，会引起深度的多大变化
    // 按照SLAM14讲的做法，不确定引起的原因包括:平移t，初始估计深度值P，然后才是像素扰动，
    // 这样做真的好吗？还是只能使用图优化的方式来实现好呢？
    const double noiseStd = 3.0;  // pixels,改为检测半径
    const Eigen::Vector3d& t = P12;
    const Eigen::Vector3d p = Pn1 * res[0];
    const Eigen::Vector3d a = p - t;
    const double alpha = acos(p.dot(t) / (p.norm() * t.norm()));
    const double belta = acos(a.dot(-t) / (a.norm() * t.norm()));
    const double deltaBelta = atan(noiseStd * max(invK(0, 0), invK(1, 1)));
    const double beltaNew = belta + deltaBelta;
    const double gamma = M_PI - alpha - beltaNew;
    const double noiseDepth = t.norm() * sin(beltaNew) / sin(gamma);

    // 新的深度及观测方差
    const double noiseIdepth = 1.0 / noiseDepth;
    const double estIdepth = 1.0 / res[0];
    const double obvCov = pow((estIdepth - noiseIdepth), 2);

    if(!RobustChi2Check(estIdepth, obvCov)) {
        return false;
    }

    // 更新均值及方差
    // cov_越小说明收敛越好，但后续需要控制其大小
    const double denominator = cov_ + obvCov;
    // 1.0-0.01 = 0.99，所以不确定范围在[1.0-0.99, 1.0+0.99]，因此直接使用逆深度即可计算
    idepth_ = (obvCov * idepth_ + cov_ * estIdepth) / denominator;
    cov_ = cov_ * obvCov / denominator;

    const double obvScov = pow((noiseDepth - res[0]), 2);
    const double sden = scov_ + obvScov;
    s_ = (obvScov * s_ + scov_ * res[0]) / sden;
    scov_ = scov_ * obvScov / sden;

    PrintDebugInfo();

    // 这里我们似乎无法计算协方差，应该说，协方差必须与运动是有关系的
    // 那么按理说，如果运动是退化的，那我们假设一个像素偏差，其实就会引起很大的不确定度，
    // 反之，当运动条件好事，单个像素所引起的不确定度应该很小，具体应该体现在上述三角化方程的稳定性上
    // obv2 = R21* K * 1/ρ1 * Pn1 + P21
    // (obv2 - P21)
    // SVO的不确定度计算：https://www.cnblogs.com/wxt11/p/7097250.html
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

void InverseDepthFilter::PrintDebugInfo() {
    const double id0 = idepth_ - sqrt(cov_), id1 = idepth_ + sqrt(cov_);
    cout << "Inverse depth info:\nidepth range: [" << id0 << ", " << idepth_
         << ", " << id1 << "], corresponding depth: [" << 1.0 / id0 << ", "
         << 1.0 / idepth_ << ", " << 1.0 / id1 << "]" << endl;

    const double d0 = s_ - sqrt(scov_), d1 = s_ + sqrt(scov_);
    cout << "Depth info:\ndepth s range: [" << d0 << ", " << s_ << ", " << d1
         << "]" << endl;
}
