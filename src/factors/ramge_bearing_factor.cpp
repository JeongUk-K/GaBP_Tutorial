#include "factors/range_bearing_factor.h"

#include "graph/factor_node.cpp"

#include <cmath>
#include <stdexcept>

namespace gsolver {

template class FactorNode<2, SE3PoseVel, SE3PoseVel>;

RangeBearingFactor::RangeBearingFactor(const std::string& id, const VectorErrorDim& measurement,
                                       const MatrixErrorDim& covariance)
    : FactorNode(id, covariance), measurement_(measurement) {
    if (!measurement.allFinite() || measurement(0) < 0.0) {
        throw std::invalid_argument("Range and bearing must be finite, with nonnegative range");
    }
    if (!covariance.allFinite() || !covariance.isApprox(covariance.transpose()) ||
        covariance.llt().info() != Eigen::Success) {
        throw std::invalid_argument("Range-bearing covariance must be finite, symmetric and positive definite");
    }
}

void RangeBearingFactor::updateErrorAndJacobian() {
    const auto& observer = edges_.at<0>()->variable_to_factor_message_.mu_.pose;
    const auto& target = edges_.at<1>()->variable_to_factor_message_.mu_.pose;
    const Eigen::Matrix3d world_to_observer = observer.linear().transpose();
    const Eigen::Vector3d point = world_to_observer * (target.translation() - observer.translation());
    const double horizontal_distance = std::hypot(point.x(), point.y());

    error_.setZero();
    J_.setZero();
    // A joint observation has no defined azimuth on the observer's z axis.
    if (horizontal_distance < 1e-9) {
        return;
    }

    const double distance = point.norm();
    const double bearing_error = std::atan2(point.y(), point.x()) - measurement_(1);
    error_ << distance - measurement_(0), std::atan2(std::sin(bearing_error), std::cos(bearing_error));

    const double horizontal_squared = horizontal_distance * horizontal_distance;
    Eigen::Matrix<double, 2, 3> measurement_jacobian;
    measurement_jacobian.row(0) = point.transpose() / distance;
    measurement_jacobian.row(1) << -point.y() / horizontal_squared, point.x() / horizontal_squared, 0.0;

    // Right perturbations express each translation increment in its own pose frame.
    J_.block<2, 3>(0, 0) = -measurement_jacobian;
    J_.block<1, 3>(1, 3) = measurement_jacobian.row(1) * skew(point);
    J_.block<2, 3>(0, SE3PoseVel::LieDim) = measurement_jacobian * world_to_observer * target.linear();
}

} // namespace gsolver
