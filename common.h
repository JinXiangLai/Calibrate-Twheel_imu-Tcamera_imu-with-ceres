#include <iostream>
#include <vector>
#include <random>

#include <ceres/ceres.h>
#include <ceres/cost_function.h>
#include <ceres/manifold.h>
#include <ceres/sized_cost_function.h>
#include <ceres/types.h>

#include <opencv2/opencv.hpp>
#include <opencv2/imgproc.hpp>  // For cv::circle
#include <opencv2/highgui.hpp> // For displaying images

#include <Eigen/Dense>

constexpr double kRad2Deg = 180 / M_PI;
constexpr double kDeg2Rad = M_PI / 180;
constexpr int kImgWidth = 640;
constexpr int kImgHeight = 480;

// 行优先存入
// clang-format off
const Eigen::Matrix3d K = (Eigen::Matrix3d() << 400.0, 0.0, 320.0,
                                                0.0, 320.0, 240.0,
                                                0.0, 0.0, 1.0).finished();
const Eigen::Matrix3d invK = K.inverse();
// clang-format on

template <typename T>
Eigen::Matrix<T, 3, 3> skew(const Eigen::Matrix<T, 3, 1> &v) {
    Eigen::Matrix<T, 3, 3> m;
    m << T(0), -v(2), v(1), v(2), T(0), -v(0), -v(1), v(0), T(0);
    return m;
}
