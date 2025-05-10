#include <iostream>
#include <vector>

#include <Eigen/Dense>

constexpr double kRad2Deg = 180 / M_PI;
constexpr double kDeg2Rad = M_PI / 180;
constexpr int kImgWidth = 640;
constexpr int kImgHeight = 480;
constexpr double kSqrt2 = 1.414213562;  // sqrt(2);

// 即条件数的相反数，主要是为了避免除以0，过小意味着协方差可信度低，
// 原因是最小奇异值很小，使得系统不稳定，易受噪声影响
constexpr double kMinReciprocalConditionNumber = 1e-10; // 1e-6

// 行优先存入
// clang-format off
constexpr double kFx = 400;
constexpr double kFy = 320;
constexpr double kCx = 320;
constexpr double kCy = 240;
const Eigen::Matrix3d K = (Eigen::Matrix3d() << kFx, 0.0, kCx,
                                                0.0, kFy, kCy,
                                                0.0, 0.0, 1.0).finished();
const Eigen::Matrix3d invK = K.inverse();

// clang-format on



// 废弃该函数
inline std::vector<int> GetEraseObservationId(const Eigen::Matrix3d& K, const Eigen::MatrixXd& A, const std::vector<Eigen::Vector3d>& Pw_cs) {
    const double fx=K.row(0)[0], cx=K.row(0)[2], fy=K.row(1)[1], cy=K.row(1)[2];
    
    const int midId = A.rows()/2-1;
    std::vector<int> keepIds, removeIds;
    keepIds.push_back(midId);

    int rowId = 0;
    while(rowId < A.rows()) {
        if (rowId == midId) {
            continue;
        }

        // 检验当前id是否值得保留
        int keep = true;
        for(int i=0; i<keepIds.size(); ++i) {
            // 检验策略：
            // 1. 因为我们已经有了运动先验，所以可以规避飞机暂停的状态
            // 2. 我们认为，若有水平运动，那么检测像素的偏差应该是较大的，所以这里就不检查水平平移量，一般水平平移量是要优先被考虑的，
            //    因为其影响W的系数，并且，当飞机距离较远时，其观测像素变化量可能较小，所以还是要独立判断tx, ty变化量
            // 3. 因此，我们仅检查像素偏差是否足够大，以确认是否可以保留
            constexpr double minHorDiff = 0.2, minPxDiff2Keep = 3.0;
            const int kr = keepIds[i];
            const double horDiff = (Pw_cs[kr/2].head(2) - Pw_cs[rowId/2].head(2)).norm();
            const Eigen::Vector2d pk{A.row(kr)[3], A.row(kr+1)[3]};
            const Eigen::Vector2d pm(A.row(rowId)[3], A.row(rowId+1)[3]);
            const Eigen::Vector2d& pnDiff = (pk-pm).cwiseAbs();
            const Eigen::Vector2d pxDiff(fx*pnDiff.x(), fy*pnDiff.y());
            // 只有水平运动足够小，且像素变化也不大的才认为是重复观测
            if(horDiff < minHorDiff && pxDiff.norm() < minPxDiff2Keep) {
                std::cout << "horDiff=" << horDiff << "& pxDiff=" << pxDiff.norm() << std::endl;
                keep = false;
                break;
            }
        }
        if(keep) {
            keepIds.push_back(rowId);
        } else {
            removeIds.push_back(rowId);
        }

        rowId += 2;
    }

    return removeIds;
}
