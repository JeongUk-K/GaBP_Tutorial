#include "multirobot/MultiRobotGraph.h"

#include <algorithm>
#include <cmath>
#include <unordered_set>
#include <utility>

namespace utias {
namespace {

template <typename T, typename Predicate>
void eraseIf(std::vector<T>& values, Predicate predicate) {
    values.erase(std::remove_if(values.begin(), values.end(), predicate), values.end());
}

void sendDampedMessages(gsolver::SE3PoseVel& pose, double weight) {
    if (pose.edges_.size() < 2) {
        return;
    }
    for (std::size_t index = 0; index < pose.edges_.size(); ++index) {
        auto& message = pose.edges_[index]->variable_to_factor_message_;
        const auto previous = message;
        pose.sendMessage(index);

        // Combine canonical parameters in the tangent frame of the current belief.
        const gsolver::SE3PoseVel::MatrixLieDim old_precision = (1.0 - weight) * previous.Lambda_;
        const gsolver::SE3PoseVel::MatrixLieDim new_precision = weight * message.Lambda_;
        const gsolver::SE3PoseVel::MatrixLieDim precision = old_precision + new_precision;
        const gsolver::SE3PoseVel::VectorLieDim eta =
            old_precision * pose.boxminus(previous.mu_) + new_precision * pose.boxminus(message.mu_);
        message.mu_ = pose.boxplus(precision.ldlt().solve(eta));
        message.Lambda_ = precision;
    }
}

} // namespace

MultiRobotGraph::MultiRobotGraph(std::size_t robot_count) : robots_(robot_count) {}

void MultiRobotGraph::addPose(RobotId robot_id, double timestamp, const Pose& pose) {
    auto& robot = robots_.at(robot_id);
    if (!pose || pose->id_.empty() || !pose->edges_.empty() || !std::isfinite(timestamp)) {
        throw std::invalid_argument("A pose with an ID, no edges and a finite timestamp is required");
    }
    if (!pose->mu_.pose.matrix().allFinite() || !pose->mu_.vel.allFinite() || !pose->Sigma_.allFinite() ||
        !pose->Sigma_.isApprox(pose->Sigma_.transpose()) || pose->Sigma_.llt().info() != Eigen::Success) {
        throw std::invalid_argument("A pose requires a finite estimate and a positive definite covariance");
    }
    if (!robot.poses.empty() && timestamp < robot.poses.back().timestamp) {
        throw std::invalid_argument("Add poses in timestamp order for each robot");
    }
    const auto [entry, inserted] = poses_.emplace(pose->id_, pose);
    if (!inserted) {
        throw std::invalid_argument("Duplicate pose ID: " + pose->id_);
    }

    const auto previous_count = robot.poses.size();
    try {
        robot.poses.push_back({timestamp, pose});
        robot.graph.addVariableNode(pose);
    } catch (...) {
        if (robot.poses.size() > previous_count) {
            robot.poses.pop_back();
        }
        poses_.erase(entry);
        throw;
    }
    pose->timestamp_ = timestamp;
}

void MultiRobotGraph::connectFactor(RobotId owner, const std::shared_ptr<gsolver::FactorNodeBase>& factor,
                                    std::vector<Pose> poses) {
    auto& graph = robots_.at(owner).graph;
    if (factor->id_.empty() || factors_.find(factor->id_) != factors_.end()) {
        throw std::invalid_argument("A factor must have a unique ID");
    }
    for (auto endpoint = poses.begin(); endpoint != poses.end(); ++endpoint) {
        const auto& pose = *endpoint;
        const auto registered = pose ? poses_.find(pose->id_) : poses_.end();
        if (registered == poses_.end() || registered->second != pose) {
            throw std::invalid_argument("Register each factor endpoint before connecting it");
        }
        if (std::find(poses.begin(), endpoint, pose) != endpoint) {
            throw std::invalid_argument("A factor cannot connect to the same pose twice");
        }
    }

    using Edge = gsolver::Edge<gsolver::SE3PoseVel>;
    std::vector<std::shared_ptr<Edge>> edges;
    edges.reserve(poses.size());
    for (const auto& pose : poses) {
        auto edge = std::make_shared<Edge>();
        edge->variable_id_ = pose->id_;
        edge->factor_id_ = factor->id_;
        edge->variable_to_factor_message_.mu_ = pose->mu_;
        edge->variable_to_factor_message_.Lambda_ = pose->Sigma_.inverse();
        edge->factor_to_variable_message_.mu_ = pose->mu_;
        edge->factor_to_variable_message_.Lambda_.setZero();
        edges.push_back(std::move(edge));
        if (pose->edges_.size() == pose->edges_.capacity()) {
            pose->edges_.reserve(2 * pose->edges_.size() + 1);
        }
    }

    const auto entry = factors_.emplace(factor->id_, FactorConnection{owner, factor, std::move(poses)}).first;
    try {
        graph.factor_nodes_.push_back(factor);
    } catch (...) {
        factors_.erase(entry);
        throw;
    }
    for (std::size_t i = 0; i < edges.size(); ++i) {
        factor->setEdgeAt(i, edges[i]);
        entry->second.poses[i]->edges_.push_back(edges[i]);
    }
}

const gsolver::FactorGraph& MultiRobotGraph::graph(RobotId robot) const {
    return robots_.at(robot).graph;
}

void MultiRobotGraph::iterate(double message_weight) {
    if (!std::isfinite(message_weight) || message_weight <= 0.0 || message_weight > 1.0) {
        throw std::invalid_argument("message_weight must be in (0, 1]");
    }
    for (const auto& robot : robots_) {
        for (const auto& factor : robot.graph.factor_nodes_) {
            factor->updateFactor();
            factor->sendAllMessages();
        }
    }
    for (const auto& robot : robots_) {
        for (const auto& pose : robot.graph.variable_nodes_) {
            pose->updateMuAndSigma();
            if (message_weight == 1.0) {
                pose->sendAllMessages();
            } else {
                sendDampedMessages(static_cast<gsolver::SE3PoseVel&>(*pose), message_weight);
            }
        }
    }
}

void MultiRobotGraph::pruneBefore(double cutoff) {
    if (!std::isfinite(cutoff)) {
        throw std::invalid_argument("The cutoff must be finite");
    }

    std::unordered_set<const VariableNodeBase*> expired_poses;
    std::unordered_set<std::string> expired_factors;
    for (const auto& robot : robots_) {
        for (const auto& timed : robot.poses) {
            if (timed.timestamp >= cutoff) {
                break;
            }
            expired_poses.insert(timed.pose.get());
            for (const auto& edge : timed.pose->edges_) {
                expired_factors.insert(edge->factor_id_);
            }
        }
    }
    if (expired_poses.empty()) {
        return;
    }

    std::unordered_set<RobotId> owners;
    std::unordered_set<gsolver::SE3PoseVel*> affected_poses;
    for (const auto& id : expired_factors) {
        const auto& connection = factors_.at(id);
        owners.insert(connection.owner);
        for (const auto& pose : connection.poses) {
            if (expired_poses.find(pose.get()) == expired_poses.end()) {
                affected_poses.insert(pose.get());
            }
        }
    }

    for (const auto& id : expired_factors) {
        const auto& connection = factors_.at(id);
        for (std::size_t i = 0; i < connection.poses.size(); ++i) {
            eraseIf(connection.poses[i]->edges_, [&](const auto& edge) { return edge->factor_id_ == id; });
            connection.factor->setEdgeAt(i, {});
        }
        factors_.erase(id);
    }
    for (const auto owner : owners) {
        eraseIf(robots_[owner].graph.factor_nodes_, [&](const auto& factor) {
            return expired_factors.find(factor->id_) != expired_factors.end();
        });
    }
    for (auto& robot : robots_) {
        if (robot.poses.empty() || robot.poses.front().timestamp >= cutoff) {
            continue;
        }
        eraseIf(robot.graph.variable_nodes_, [&](const auto& pose) {
            return expired_poses.find(pose.get()) != expired_poses.end();
        });
        while (!robot.poses.empty() && robot.poses.front().timestamp < cutoff) {
            poses_.erase(robot.poses.front().pose->id_);
            robot.poses.pop_front();
        }
    }

    for (auto* pose : affected_poses) {
        if (pose->edges_.empty()) {
            // Match the covariance of an unconstrained gsolver belief with its 1e-6 precision regularization.
            pose->Sigma_.setIdentity();
            pose->Sigma_ *= 1e6;
            continue;
        }
        pose->updateMuAndSigma();
        if (pose->edges_.size() == 1) {
            // gsolver leaves the outgoing message untouched at a leaf.
            auto& message = pose->edges_.front()->variable_to_factor_message_;
            message.mu_ = pose->mu_;
            message.Lambda_.setZero();
        } else {
            pose->sendAllMessages();
        }
    }
}

} // namespace utias
