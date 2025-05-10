#include <Eigen/Core>

#include <opencv2/highgui.hpp>  // For displaying images
#include <opencv2/imgproc.hpp>  // For cv::circle
#include <opencv2/opencv.hpp>

#include "classes.h"

template <typename T>
Eigen::Matrix<T, 3, 3> skew(const Eigen::Matrix<T, 3, 1>& v) {
    Eigen::Matrix<T, 3, 3> m;
    m << T(0), -v(2), v(1), v(2), T(0), -v(0), -v(1), v(0), T(0);
    return m;
}

Eigen::Vector3d CrossProduct(const Eigen::Vector3d& a,
                             const Eigen::Vector3d& b);

Eigen::Vector3d TransformPw2Pc(const Eigen::Matrix3d& Rc_w,
                               const Eigen::Vector3d& Pc_w,
                               const Eigen::Vector3d& Pw);

Eigen::Vector2d ProjectPc2PixelPlane(const Eigen::Vector3d& K,
                                     const Eigen::Vector3d& Pc);

Eigen::Vector2d ProjectPw2PixelPlane(const Eigen::Matrix3d& Rc_w,
                                     const Eigen::Vector3d& Pc_w,
                                     const Eigen::Vector3d& K,
                                     const Eigen::Vector3d& Pw);

Eigen::Matrix<double, 2, 3> CalculateObvWrtPwJacobian(
    const Eigen::Matrix3d& Rc_w, const Eigen::Vector3d& Pc_w,
    const Eigen::Matrix3d& K, const Eigen::Vector3d& Pw);

Eigen::Matrix3d CalculateHessianMatrix(
    const std::deque<DataFrame>& slidingWindow, const Eigen::Matrix3d& K,
    const Eigen::Vector3d& Pw);

bool CalculateCovariance(const Eigen::Matrix3d& H, Eigen::Matrix3d& cov,
                         const double& sigma2 = 9);

std::vector<Eigen::Vector3d> TransformPw2Pc(
    const std::vector<Eigen::Matrix3d>& Rc_ws,
    const std::vector<Eigen::Vector3d>& Pc_ws, const Eigen::Vector3d& Pw);

void GenerateNextPose(const Eigen::Matrix3d& Rw_c1,
                      const Eigen::Vector3d& Pw_c1,
                      const Eigen::Vector3d& rotAxis,
                      const Eigen::Vector3d& posDirection, const double& rotAng,
                      const double& moveDist, Eigen::Matrix3d& Rw_c2,
                      Eigen::Vector3d& Pw_c2, const bool isDeg = true);

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
    const ::Eigen::Matrix3d& K, Eigen::Vector4d& singularValues);

Eigen::Matrix3d RPY2Rotation(const Eigen::Vector3d& _rpy,
                             const bool isDeg = true);

Eigen::Vector2d ProjectPw2Pixel(const Eigen::Vector3d& Pw,
                                const Eigen::Matrix3d& Rcw,
                                const Eigen::Vector3d& Pcw,
                                const Eigen::Matrix3d& K);

void CalculateInitialPwDLT(const std::deque<DataFrame>& slidingWindow,
                           const Eigen::Matrix3d& K, Eigen::Vector3d& initPw,
                           Eigen::Vector4d& singularValues);

bool CheckInitialPwValidity(const std::vector<Eigen::Matrix3d>& Rc_w,
                            const std::vector<Eigen::Vector3d>& Pc_w,
                            const Eigen::Vector3d& initPw);

std::vector<int> GetEraseObservationId(
    const std::deque<DataFrame>& slidingWindow);

void ExtrackPoseAndObvFromSlidingWindow(
    const std::deque<DataFrame>& slidingWindow,
    std::vector<Eigen::Matrix3d>& Rc_ws, std::vector<Eigen::Vector3d>& Pc_ws,
    std::vector<std::vector<Eigen::Vector2d>>& obvs);

/**
 * @brief 拼接多张图像并绘制观测点匹配连线
 * @param debugImgs 输入的图像列表（每个相机视角一张图）
 * @param obvs 观测点列表（每个视角对应的观测像素坐标）
 * @return cv::Mat 拼接后的图像
 */
cv::Mat stitchAndDrawMatches(
    const std::deque<cv::Mat>& debugImgs,
    const std::deque<std::vector<Eigen::Vector2d>>& obvs);

cv::Mat stitchAndDrawMatches(const std::deque<DataFrame>& slidingWindow);

// 直接利用对地高度计算初值
Eigen::Vector3d CalculatePriorPwByHeight(
    const Eigen::Matrix3d& Rwc, const Eigen::Vector3d& Pwc,
    const double& horizontalHeight, const Eigen::Matrix3d& K,
    const std::vector<Eigen::Vector2d>& obvs);

void CullingBadObservationsBeforeInit(std::deque<DataFrame>& slidingWindow);

bool CheckLastestObservationUseful(std::deque<DataFrame>& slidingWindow);

Eigen::Vector3d LogSO3(const Eigen::Matrix3d& R);

Eigen::Matrix3d InverseRightJacobianSO3(const double x, const double y,
                                        const double z);

Eigen::Matrix3d InverseRightJacobianSO3(const Eigen::Vector3d& v);

void PrintReprojectErrorEachFrame(const std::deque<DataFrame>& sw,
                                  const Eigen::Vector3d& Pw,
                                  const Eigen::Matrix3d& K);

double CalculateChi2Distance(const Eigen::Matrix3d& cov,
                             const Eigen::Vector3d& pos,
                             const Eigen::Vector3d& est);
