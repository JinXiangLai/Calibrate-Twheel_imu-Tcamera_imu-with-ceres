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

inline Eigen::Vector3d TransformPw2Pc(const Eigen::Matrix3d& Rc_w,
                                      const Eigen::Vector3d& Pc_w,
                                      const Eigen::Vector3d& Pw) {
    const Eigen::Matrix<double, 3, 1> Pc = Rc_w * Pw + Pc_w;
    return Pc;
}

inline Eigen::Vector2d ProjectPc2PixelPlane(const Eigen::Vector3d& K,
                                            const Eigen::Vector3d& Pc) {
    const Eigen::Matrix<double, 3, 1> Pn = Pc / Pc[2];
    const Eigen::Matrix<double, 2, 1> obv = K.block(0, 0, 2, 3) * Pn;
    return obv;
}

inline Eigen::Vector2d ProjectPw2PixelPlane(const Eigen::Matrix3d& Rc_w,
                                            const Eigen::Vector3d& Pc_w,
                                            const Eigen::Vector3d& K,
                                            const Eigen::Vector3d& Pw) {
    const Eigen::Vector3d& Pc = TransformPw2Pc(Rc_w, Pc_w, Pw);
    const Eigen::Vector2d& obv = ProjectPc2PixelPlane(K, Pc);
    return obv;
}

inline Eigen::Matrix<double, 2, 3> CalculateObvWrtPwJacobian(
    const Eigen::Matrix3d& Rc_w, const Eigen::Vector3d& Pc_w,
    const Eigen::Vector3d& K, const Eigen::Vector3d& Pc) {

    const Eigen::Matrix<double, 2, 3> J_r_Pn = K.block(0, 0, 2, 3);
    const double invZ = 1 / Pc[2];
    const double invZ2 = invZ * invZ;
    // clang-format off
    const Eigen::Matrix3d J_Pn_Pc =
        (Eigen::Matrix3d() << invZ, 0, -Pc[0] * invZ2, 
                              0, invZ, -Pc[1] * invZ2,
                              0, 0, 0).finished();
    // clang-format on
    const Eigen::Matrix<double, 2, 3> j = J_r_Pn * J_Pn_Pc * Rc_w;
    return j;
}

inline std::vector<Eigen::Vector3d> TransformPw2Pc(
    const std::vector<Eigen::Matrix3d>& Rc_ws,
    const std::vector<Eigen::Vector3d>& Pc_ws, const Eigen::Vector3d& Pw) {
    std::vector<Eigen::Vector3d> Pcs(Rc_ws.size());
    for (int i = 0; i < Rc_ws.size(); ++i) {
        Pcs[i] = TransformPw2Pc(Rc_ws[i], Pc_ws[i], Pw);
    }
    return Pcs;
}


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
              const double& _t, const cv::Mat& img)
        : Rc_w(_Rc_w),
          Pc_w(_Pc_w),
          height2Ground(height),
          obv(_obv),
          timestamp(_t),
          debugImg(img) {
        return;
    }

    Eigen::Vector3d GetPw() const { return -Rc_w.transpose() * Pc_w; }
    Eigen::Vector2d GetMainObv() const {return obv[0];}

    double timestamp = 0.;
    Eigen::Matrix3d Rc_w = Eigen::Matrix3d::Identity();
    Eigen::Vector3d Pc_w = Eigen::Vector3d::Zero();
    double height2Ground = 0.;
    std::vector<Eigen::Vector2d> obv;
    cv::Mat debugImg;
};

class PriorEstimateData {
   public:
        void UpdateMean(const Eigen::Vector3d& p) {
            const Eigen::Vector3d sum = updateCount_ * Pw_ + p;
            ++updateCount_;
            Pw_ = sum / updateCount_;
            historyEstPw_.emplace_back(p);
        }
        void UpdateMessageMatrix(const Eigen::Matrix3d& cov) {
            Eigen::Matrix3d C = cov;
            const double ratio = C.diagonal().norm() / (3 * 0.15);
            if(ratio < 1) {
                C.diagonal() /= ratio;
            }
            //C.diagonal() += Eigen::Vector3d::Ones() * 1e-15;
            const Eigen::Matrix3d& h = C.inverse();
            H_ += h;
            historyEstCov_.emplace_back(cov);
        }
        void UpdateMeanAndH(const Eigen::Vector3d& p, const Eigen::Matrix3d& cov) {
            UpdateMean(p);
            UpdateMessageMatrix(cov);
        }
        Eigen::Vector3d GetPw() const {return Pw_;}
        Eigen::Matrix3d GetH() const {return H_;}
   private:
    Eigen::Vector3d Pw_ = Eigen::Vector3d::Zero();
    Eigen::Matrix3d H_ = Eigen::Matrix3d::Zero();
    int updateCount_ = 0;

    // 不一定需要的变量
    std::vector<Eigen::Vector3d> historyEstPw_;
    std::vector<Eigen::Matrix3d> historyEstCov_;
};
