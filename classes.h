#pragma once
#include <vector>

#include <opencv2/opencv.hpp>

#include <Eigen/Dense>  // 仅具有矩阵运算基本的定义及基础计算

struct FrameData {
    FrameData(const Eigen::Matrix3d& _Rc_w, const Eigen::Vector3d& _Pc_w,
              const double& height, const std::vector<Eigen::Vector2d>& _obv,
              const double& _t, const cv::Mat& img);

    Eigen::Vector3d GetPw() const { return -Rc_w.transpose() * Pc_w; }
    Eigen::Vector3d GetPc(const Eigen::Vector3d& Pw) const {
        return Rc_w * Pw + Pc_w;
    }
    Eigen::Vector2d GetMainObv() const { return obv[0]; }
    // TODO: 可以根据残差去剔除一些没有收敛的观测以期提高精度
    Eigen::Vector2d GetObvResidual(const Eigen::Vector3d& Pw,
                                   const Eigen::Matrix3d& K) const;
    //Eigen::Vector2d GetObvResidual(const Eigen::Vector3d& Pc){
    //    const Eigen::Vector2d Pn = Pc/Pc.z()
    //}
    double GetNormObvResidual(const Eigen::Vector3d& Pw) const;

    double timestamp = 0.;
    Eigen::Matrix3d Rc_w = Eigen::Matrix3d::Identity();
    Eigen::Vector3d Pc_w = Eigen::Vector3d::Zero();
    double height2Ground = 0.;
    std::vector<Eigen::Vector2d> obv;
    cv::Mat debugImg;
    Eigen::Vector2d obvNorm;
};

// 不再被作为先验约束
class PriorEstimateData {
   public:
    void UpdateMean(const Eigen::Vector3d& p);
    void UpdateMessageMatrix(const Eigen::Matrix3d& cov);
    void UpdateMeanAndH(const Eigen::Vector3d& p, const Eigen::Matrix3d& cov);
    Eigen::Vector3d GetPw() const { return Pw_; }
    Eigen::Matrix3d GetH() const { return H_; }

    Eigen::Vector3d lastPw_ = Eigen::Vector3d(0, 0, 0);
    Eigen::Matrix3d lastCov_ = Eigen::Matrix3d::Zero();

   private:
    Eigen::Vector3d Pw_ = Eigen::Vector3d::Zero();
    Eigen::Matrix3d H_ = Eigen::Matrix3d::Zero();
    int updateCount_ = 0;

    // 不一定需要的变量
    std::vector<Eigen::Vector3d> historyEstPw_;
    std::vector<Eigen::Matrix3d> historyEstCov_;
};

class Pose {
   public:
    Pose(const Eigen::Quaterniond& q_wb, const Eigen::Vector3d& p_wb)
        : q_wb_(q_wb), p_wb_(p_wb) {}
    Pose(const Eigen::Vector3d& rpy, const Eigen::Vector3d& pos,
         const bool isAngle = true)
        : p_wb_(pos) {
        Eigen::Vector3d r = rpy;
        if (isAngle) {
            r = rpy * M_PI / 180;
        }
        q_wb_ = Eigen::AngleAxisd(r[2], Eigen::Vector3d::UnitZ()) *
                Eigen::AngleAxisd(r[1], Eigen::Vector3d::UnitY()) *
                Eigen::AngleAxisd(r[0], Eigen::Vector3d::UnitX());
    }
    Pose operator*(const Pose& T) {
        Eigen::Quaterniond q = q_wb_ * T.q_wb_;
        Eigen::Vector3d p = q_wb_ * T.p_wb_ + p_wb_;
        return Pose(q, p);
    }
    Pose Inverse() { return Pose(q_wb_.inverse(), -(q_wb_.inverse() * p_wb_)); }
    Eigen::Quaterniond q_wb_;
    Eigen::Vector3d p_wb_;
};

struct InverseDepthFilter {
    InverseDepthFilter(const double& idepth, const double& depthMin,
                       const double& depthMax, const FrameData& curF);

    InverseDepthFilter(const FrameData& curF);

    bool Update(const FrameData& curF);

    // 鲁棒性更新
    bool UpdateWithRobustCheck(const double& idepthObs,
                               const double& obsNoisepixel,
                               const double& baseline);

    // 第一次计算，后续ρ1已知可按EKF方法估计
    bool UpdateInverseDepth(const FrameData& curF, const Eigen::Matrix3d& invK);

    // 坐标系变换（带协方差传播）
    bool TransformHost(const FrameData& curF, const Eigen::Matrix3d& invK);

    double idepth_ = 1.0;
    double cov_ = 0.08;  // 默认初始值
    bool initialized_ = false;
    FrameData host_;
    FrameData last_;
};
