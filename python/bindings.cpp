#include "factors/range_bearing_factor.h"
#include "factors/range_factor.h"
#include "multirobot/MultiRobotGraph.h"

#include <gsolver/graph/types/factors/se3_posevel_posevel.h>
#include <gsolver/graph/types/factors/se3_posevel_prior.h>

#include <nanobind/eigen/dense.h>
#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/shared_ptr.h>
#include <nanobind/stl/string.h>

#include <new>

namespace nb = nanobind;

NB_MODULE(gabp, m) {
    nb::module_::import_("gsolver");

    using gsolver::FactorNodeBase;
    using gsolver::RangeBearingFactor;
    using gsolver::RangeFactor;
    using gsolver::SE3PoseVel;
    using gsolver::SE3PoseVelPoseVel;
    using gsolver::SE3PoseVelPrior;
    using utias::MultiRobotGraph;
    using namespace nb::literals;

    nb::class_<RangeFactor, FactorNodeBase>(m, "RangeFactor")
        .def("__init__", [](RangeFactor* self, const std::string& id, double range,
                            nb::ndarray<nb::numpy, const double, nb::shape<1, 1>, nb::device::cpu> covariance) {
            new (self) RangeFactor(id, range, RangeFactor::MatrixErrorDim::Constant(covariance(0, 0)));
        }, "id"_a, "range"_a, "covariance"_a);

    nb::class_<RangeBearingFactor, FactorNodeBase>(m, "RangeBearingFactor")
        .def(nb::init<const std::string&, const RangeBearingFactor::VectorErrorDim&,
                      const RangeBearingFactor::MatrixErrorDim&>(),
             "id"_a, "measurement"_a, "covariance"_a);

    nb::class_<MultiRobotGraph>(m, "MultiRobotGraph")
        .def(nb::init<std::size_t>(), "robot_count"_a)
        .def("add_pose", &MultiRobotGraph::addPose, "robot"_a, "timestamp"_a, "pose"_a)
        .def("add_factor", &MultiRobotGraph::addFactor<SE3PoseVelPrior, SE3PoseVel>,
             "owner"_a, "factor"_a, "pose"_a)
        .def("add_factor", &MultiRobotGraph::addFactor<SE3PoseVelPoseVel, SE3PoseVel, SE3PoseVel>,
             "owner"_a, "factor"_a, "from_pose"_a, "to_pose"_a)
        .def("add_factor", &MultiRobotGraph::addFactor<RangeFactor, SE3PoseVel, SE3PoseVel>,
             "owner"_a, "factor"_a, "observer"_a, "target"_a)
        .def("add_factor", &MultiRobotGraph::addFactor<RangeBearingFactor, SE3PoseVel, SE3PoseVel>,
             "owner"_a, "factor"_a, "observer"_a, "target"_a)
        .def("iterate", &MultiRobotGraph::iterate, "message_weight"_a = 1.0)
        .def("prune_before", &MultiRobotGraph::pruneBefore, "cutoff"_a);
}
