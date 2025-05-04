// 研究系数矩阵量级与条件数、奇异值、解精度的关系

#include "common.hpp"

#include "utils.h"

using namespace std;

int main() {
    const double x = 3, y = 4;  // 目标解
    const double noise = 0.1, estX = x + noise; // 最优解有噪声很正常

    // 退化系数矩阵
    Eigen::Matrix2d A0;
    A0 << 10, 1.1, 30.1, 2.0;
    Eigen::Vector2d b0;
    // x被错误估计为3.1是很正常的
    b0 << x * A0.row(0)[0] + y * A0.row(0)[1], // 这里使用扰动的x，相当于等式右边加了一点噪声
        estX * A0.row(1)[0] + y * A0.row(1)[1]; // 这里不使用扰动的x
    const Eigen::Vector2d res0 = A0.colPivHouseholderQr().solve(b0);
    Eigen::JacobiSVD<Eigen::Matrix2d> svd0(A0, Eigen::ComputeFullV);
    cout << "A0 singular value: " << svd0.singularValues().transpose() << endl;
    cout << "condition number 0: " << svd0.singularValues()[0]/svd0.singularValues()[1] << endl;
    cout << "res0: " << res0.transpose() << endl << endl;

    Eigen::Matrix2d A1;
    A1 << 1, 1.1, 1, 2.0;
    Eigen::Vector2d b1;
    b1 << x * A1.row(0)[0] + y * A1.row(0)[1],
        estX * A1.row(1)[0] + y * A1.row(1)[1];
    const Eigen::Vector2d res1 = A1.colPivHouseholderQr().solve(b1);
    Eigen::JacobiSVD<Eigen::Matrix2d> svd1(A1, Eigen::ComputeFullV);
    cout << "A1 singular value: " << svd1.singularValues().transpose() << endl;
    cout << "condition number 1: " << svd1.singularValues()[0]/svd1.singularValues()[1] << endl;
    cout << "res1: " << res1.transpose() << endl << endl;

    const Eigen::Vector3d a(1, 2, 3), b(4, 5, 6);
    const Eigen::Vector3d cr0 = a.cross(b);
    const Eigen::Vector3d cr1 = CrossProduct(a, b);
    cout << "cr0: " << cr0.transpose() << endl;;
    cout << "cr1: " << cr1.transpose() << endl;
}