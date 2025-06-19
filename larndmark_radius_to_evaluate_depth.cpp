// 利用已知marker的半径，来估计深度
#include <random>

#include "common.hpp"

using namespace std;

// 实际偏差会很大，该方法不可行
constexpr double kPreSetRadius = 5.0;  // 预写入的半径参数，实际可能存在出入

int main(int argc, char** argv) {
    // 注意：这个深度不是在相机系下的，不能误认为是相机系下，即便所有世界点在同一个Zw=depth平面上，
    // 其转换到相机坐标系时，就不是在Zc=depth平面上了
    double depth = 50.0;
    double realRadius = 3.0;  // 真实marker对应的半径，检测误差也考虑在内

    if (argc > 1) {
        depth = atof(argv[1]);
    }
    if (argc > 2) {
        realRadius = atof(argv[2]);
    }

    printf("depth=%f, realRadius=%f\n", depth, realRadius);

    // 产生随机数计算两个真实世界点，即圆心和半径上一点在世界系下的真实位置
    mt19937 gen1(42);
    uniform_real_distribution<double> uniformDist(0, 100);
    const Eigen::Vector3d Pw0(uniformDist(gen1), uniformDist(gen1), depth);
    // 随机产生一个方向向量
    const Eigen::Vector2d directions(uniformDist(gen1), uniformDist(gen1));
    Eigen::Vector3d Pw1 = Pw0;
    Pw1.head(2) += directions.normalized() * realRadius;

    // 两个地图点投影到同一个图像上，图像位置固定为(0, 0, 0)，仅设定姿态
    Eigen::Vector3d Pc_w(0, 0, 0);
    // TODO：必须要检验飞机的姿态角，才可以使用其对地高度作为相机到水平面的高度！！！
    Eigen::Vector3d rpy(0, 0,
                        0.);  // 一般情况下，飞机只会有绕Z轴旋转，否则就翻车了
    rpy *= kDeg2Rad;
    // 相机坐标系下：X-pitch, Y-yaw, Z-roll
    Eigen::Matrix3d Rc_w =
        Eigen::AngleAxisd(rpy[2], Eigen::Vector3d::UnitZ()) *
        Eigen::AngleAxisd(rpy[1], Eigen::Vector3d::UnitY()) *
        Eigen::AngleAxisd(rpy[0], Eigen::Vector3d::UnitX()).toRotationMatrix();

    // 计算两个地图点的实际投影像素
    const Eigen::Vector3d Pc0 = Rc_w * Pw0 + Pc_w;
    const Eigen::Vector3d Pc1 = Rc_w * Pw1 + Pc_w;
    cout << "(Pc1-Pc0).norm: " << (Pc1 - Pc0).norm() << endl;
    const double Zc0 = Pc0.z();
    const double Zc1 = Pc1.z();
    printf("Zc0=%f, Zc1=%f\n", Zc0, Zc1);
    // 获取齐次像素坐标
    const Eigen::Vector3d Pn0 = Pc0 / Pc0.z();
    const Eigen::Vector3d Pn1 = Pc1 / Pc1.z();
    Eigen::Vector3d px0 = K * Pn0;
    Eigen::Vector3d px1 = K * Pn1;
    cout << "px0: " << px0.transpose() << endl;
    cout << "px1: " << px1.transpose() << endl;

    // OK，现在利用CNN能够给出像素偏差了，其被认为对应真实世界的5m
    const double radius = (px1 - px0).norm();  // 像素偏差，即测量的像素半径
    printf("measurement radius=%f pixels.\n\n", radius);

    // 我们能够获取marker所在平面上每个像素代表的物理长度
    const double cx=K(0, 2), cy=K(1, 2);
    const double distPerPixel = kPreSetRadius / radius;
    const double predictXc0 = (px0.x() - cx) * distPerPixel, predictYc0 = (px0.y() - cy) * distPerPixel;
    // Xn0 = Xc0/Zc0, Yn0 = Yc0/Zc0 
    const double predictZc0 =(predictXc0/Pn0.x() + predictYc0/Pn0.y()) * 0.5;

    const double predictXc1 = (px1.x() - cx) * distPerPixel, predictYc1 = (px1.y() - cy) * distPerPixel;
    // Xn0 = Xc0/Zc0, Yn0 = Yc0/Zc0 
    const double predictZc1 =(predictXc0/Pn1.x() + predictYc0/Pn1.y()) * 0.5;
    
//     // 接下来解算深度，根据逆投影规则有：
//     // Pc0 = d0 * invK * px0
//     // Pc1 = d1 * invK * px1
//     // (Pc1-Pc0) = invK * (d1*px1- d0*px0)
//     // 上式有个bug，即当Pc1.z == Pc0.z时，左式只有2个有效约束，第3个为0,是无效约束
//     // 当利用位姿判断飞机平稳飞行时，有d0=d1，则有：
//     // d = (invK  *Δpx).T * ΔPc / (invK  *Δpx).T * (invK * Δpx)...(4)
//     // 需注意：世界系下是同一个平面，不能保证相机系下是同一个平面，除非相机只有绕Z轴的转动
//     //const Eigen::Vector3d deltaPc = Pc1 - Pc0; // 这个是未知的，且只有模长被认为已知，且设为 kPreSetRadius
//     // 因此无法根据试(4)求解d，只能通过相似三角形按比例求解，即:
//     // 1/deltaInNorm = d/kPreSetRadius
//     Eigen::Vector3d deltaPx(0, 0, 0);
//     deltaPx = (px1 - px0);

//     const Eigen::Vector3d deltaPn = invK * deltaPx;
//     const Eigen::Vector2d usefulConstraint = deltaPn.head(2);
//     cout << "usefulConstraint norm in normalPlane: " << usefulConstraint.norm() << endl;

// #define USE_TRUE_DEPTH 0

// #if USE_TRUE_DEPTH
//     const double estDepth = 50;  // 即便给出真实深度也不能算出正确值？
// #else
//     // 该等式只有在飞机只有绕相机Z轴旋转时成立
//     const double deltaLengthInNormalPlane = deltaPn.norm();
//     const double estDepth = kPreSetRadius / deltaLengthInNormalPlane;
// #endif

    // const Eigen::Vector3d estPc0 = estDepth * invK * px0;
    // const Eigen::Vector3d estPc1 = estDepth * invK * px1;

    const Eigen::Vector3d estPc0(predictXc0, predictYc0, predictZc0);
    const Eigen::Vector3d estPc1(predictXc1, predictYc1, predictZc1);
    cout << "(estPc1-estPc0).norm: " << (estPc1 - estPc0).norm() << endl;

    // 打印结果
    // printf("trueDepth=%f, estDepth=%f\n", depth, estDepth);
    cout << "truePc0: " << Pc0.transpose() << endl;
    cout << "estPc0: " << estPc0.transpose() << endl << endl;
    cout << "truePc1: " << Pc1.transpose() << endl;
    cout << "estPc1: " << estPc1.transpose() << endl;

    return 0;
}