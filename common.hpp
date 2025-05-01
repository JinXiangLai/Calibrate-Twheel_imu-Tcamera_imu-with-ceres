#include <deque>
#include <iostream>
#include <random>
#include <vector>

#include <ceres/ceres.h>
#include <ceres/cost_function.h>
#include <ceres/manifold.h>
#include <ceres/sized_cost_function.h>
#include <ceres/types.h>

#include <opencv2/highgui.hpp>  // For displaying images
#include <opencv2/imgproc.hpp>  // For cv::circle
#include <opencv2/opencv.hpp>

#include <Eigen/Dense>

constexpr double kRad2Deg = 180 / M_PI;
constexpr double kDeg2Rad = M_PI / 180;
constexpr int kImgWidth = 640;
constexpr int kImgHeight = 480;
constexpr double kSqrt2 = 1.414213562;  // sqrt(2);


// 行优先存入
// clang-format off
const Eigen::Matrix3d K = (Eigen::Matrix3d() << 400.0, 0.0, 320.0,
                                                0.0, 320.0, 240.0,
                                                0.0, 0.0, 1.0).finished();
const Eigen::Matrix3d invK = K.inverse();

template <typename T>
Eigen::Matrix<T, 3, 3> skew(const Eigen::Matrix<T, 3, 1>& v) {
    Eigen::Matrix<T, 3, 3> m;
    m << T(0), -v(2), v(1), 
         v(2), T(0), -v(0), 
         -v(1), v(0), T(0);
    return m;
}

inline Eigen::Vector3d CrossProduct(const Eigen::Vector3d& a, const Eigen::Vector3d& b) {
    return Eigen::Vector3d{
        a[1]*b[2] - a[2]*b[1],
        a[2]*b[0] - a[0]*b[2],
        a[0]*b[1] - a[1]*b[0]
    };
}
// clang-format on


inline void GenerateNextPose(const Eigen::Matrix3d& Rw_c1,
                             const Eigen::Vector3d& Pw_c1,
                             const Eigen::Vector3d& rotAxis,
                             const Eigen::Vector3d& posDirection,
                             const double& rotAng, const double& moveDist,
                             Eigen::Matrix3d& Rw_c2, Eigen::Vector3d& Pw_c2,
                             const bool isDeg = true) {
    const double rad = isDeg ? rotAng * kDeg2Rad : rotAng;
    const Eigen::Matrix3d& Rc1_c2 =
        Eigen::AngleAxisd(rad, rotAxis.normalized()).toRotationMatrix();
    Rw_c2 = Rw_c1 * Rc1_c2;
    const Eigen::Vector3d Pc1_c2 = moveDist * posDirection.normalized();
    Pw_c2 = Rw_c1 * Pc1_c2 + Pw_c1;
}

struct DataFrame {
    DataFrame(const Eigen::Matrix3d& _Rc_w, const Eigen::Vector3d& _Pc_w,
              const double& height, const std::vector<Eigen::Vector2d>& _obv,
              const double& _t)
        : Rc_w(_Rc_w),
          Pc_w(_Pc_w),
          height2Ground(height),
          obv(_obv),
          timestamp(_t) {
        return;
    }

    Eigen::Vector3d GetPw() const {return -Rc_w.transpose() * Pc_w;}

    double timestamp = 0.;
    Eigen::Matrix3d Rc_w = Eigen::Matrix3d::Identity();
    Eigen::Vector3d Pc_w = Eigen::Vector3d::Zero();
    double height2Ground = 0.;
    std::vector<Eigen::Vector2d> obv;
};
