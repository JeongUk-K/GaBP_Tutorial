#pragma once

#include <gsolver/graph/factor_graph.h>
#include <gsolver/graph/types/variables/se3_posevel.h>

#include <cstddef>
#include <deque>
#include <memory>
#include <stdexcept>
#include <string>
#include <tuple>
#include <type_traits>
#include <unordered_map>
#include <vector>

namespace utias {

class MultiRobotGraph {
public:
    using RobotId = std::size_t;
    using Pose = std::shared_ptr<gsolver::SE3PoseVel>;

    // Robot IDs are zero-based indices. Pose and factor IDs are unique across the graphs.
    explicit MultiRobotGraph(std::size_t robot_count);
    MultiRobotGraph(const MultiRobotGraph&) = delete;
    MultiRobotGraph& operator=(const MultiRobotGraph&) = delete;

    // Use the supplied timestamp for both the node and the sliding window.
    void addPose(RobotId robot, double timestamp, const Pose& pose);

    // Pass endpoints in the order the factor expects, including endpoints owned by another robot.
    template <typename Factor, typename... Poses>
    void addFactor(RobotId owner, const std::shared_ptr<Factor>& factor, const std::shared_ptr<Poses>&... poses) {
        static_assert(sizeof...(Poses) > 0 && (std::is_same_v<Poses, gsolver::SE3PoseVel> && ...));
        static_assert(std::is_same_v<typename Factor::VariableTypes, std::tuple<Poses...>>);
        if (!factor) {
            throw std::invalid_argument("A factor is required");
        }
        for (std::size_t i = 0; i < sizeof...(Poses); ++i) {
            if (factor->edges_.getEdgeAt(i)) {
                throw std::invalid_argument("The factor is already connected");
            }
        }
        connectFactor(owner, factor, {poses...});
    }

    const gsolver::FactorGraph& graph(RobotId robot) const;
    void iterate(double message_weight = 1.0);
    // Remove poses strictly older than cutoff and detach every incident factor.
    void pruneBefore(double cutoff);

private:
    struct TimedPose {
        double timestamp;
        Pose pose;
    };

    struct RobotGraph {
        gsolver::FactorGraph graph;
        std::deque<TimedPose> poses;
    };

    struct FactorConnection {
        RobotId owner;
        std::shared_ptr<gsolver::FactorNodeBase> factor;
        std::vector<Pose> poses;
    };

    void connectFactor(RobotId owner, const std::shared_ptr<gsolver::FactorNodeBase>& factor,
                       std::vector<Pose> poses);

    std::vector<RobotGraph> robots_;
    std::unordered_map<std::string, Pose> poses_;
    std::unordered_map<std::string, FactorConnection> factors_;
};

} // namespace utias
