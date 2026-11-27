#include "factors/range_factor.h"
#include "factors/range_bearing_factor.h"

#include <gsolver/graph/helpers/factor_graph_helper.h>

#include "graph/factor_node.cpp"

#include <cmath>
#include <stdexcept>

namespace gsolver {

template class FactorNode<1, SE3PoseVel, SE3PoseVel>;

RangeFactor::RangeFactor(const std::string& id, double range, const MatrixErrorDim& covariance)
    : FactorNode(id, covariance), range_(range) {
    if (!std::isfinite(range) || range < 0.0) {
        throw std::invalid_argument("Range must be finite and nonnegative");
    }
    if (!covariance.allFinite() || covariance(0, 0) <= 0.0) {
        throw std::invalid_argument("Range variance must be finite and positive");
    }
}

void RangeFactor::updateErrorAndJacobian() {
    const auto& observer = edges_.at<0>()->variable_to_factor_message_.mu_.pose;
    const auto& target = edges_.at<1>()->variable_to_factor_message_.mu_.pose;
    const Eigen::Vector3d delta = target.translation() - observer.translation();
    const double distance = delta.norm();

    error_(0) = distance - range_;
    J_.setZero();
    if (distance > 0.0) {
        const Eigen::RowVector3d direction = delta.transpose() / distance;
        J_.block<1, 3>(0, 0) = -direction * observer.linear();
        J_.block<1, 3>(0, SE3PoseVel::LieDim) = direction * target.linear();
    }
}

namespace {

std::vector<std::shared_ptr<VariableNodeBase>> findVariables(FactorGraph& graph,
                                                            const std::vector<std::string>& ids) {
    std::vector<std::shared_ptr<VariableNodeBase>> variables;
    variables.reserve(ids.size());
    for (const auto& id : ids) {
        const auto variable = graph.getVariableNode(id);
        if (!variable) {
            throw std::invalid_argument("Unknown variable ID: " + id);
        }
        variables.push_back(variable);
    }
    return variables;
}

// gsolver compiles addFactorNode only for its built-in factor types.
template <typename Factor>
void connectObservation(FactorGraph& graph, const std::shared_ptr<Factor>& factor,
                        const std::vector<std::shared_ptr<VariableNodeBase>>& variables) {
    if (!factor || variables.size() != Factor::NumberOfEdges_) {
        throw std::invalid_argument("An observation factor requires two pose nodes");
    }
    if (!std::dynamic_pointer_cast<SE3PoseVel>(variables[0]) || !std::dynamic_pointer_cast<SE3PoseVel>(variables[1])) {
        throw std::invalid_argument("An observation factor requires two SE3PoseVel nodes");
    }
    if (variables[0]->id_ == variables[1]->id_) {
        throw std::invalid_argument("The observer and target must have distinct IDs");
    }
    if (factor->edges_.template at<0>() || factor->edges_.template at<1>()) {
        throw std::invalid_argument("The observation factor is already connected");
    }

    using GraphHelper = typename FactorGraphHelperBuilder<Factor, typename Factor::VariableTypes>::type;
    GraphHelper::createEdge(0, factor, variables[0]);
    GraphHelper::createEdge(1, factor, variables[1]);
    graph.factor_nodes_.push_back(factor);
}

} // namespace

template <>
void FactorGraph::addFactorNode(const std::shared_ptr<RangeFactor>& factor, std::vector<std::string> variable_ids) {
    addFactorNode(factor, findVariables(*this, variable_ids));
}

template <>
void FactorGraph::addFactorNode(const std::shared_ptr<RangeFactor>& factor,
                                std::vector<std::shared_ptr<VariableNodeBase>> variables) {
    connectObservation(*this, factor, variables);
}

template <>
void FactorGraph::addFactorNode(const std::shared_ptr<RangeBearingFactor>& factor,
                                std::vector<std::string> variable_ids) {
    addFactorNode(factor, findVariables(*this, variable_ids));
}

template <>
void FactorGraph::addFactorNode(const std::shared_ptr<RangeBearingFactor>& factor,
                                std::vector<std::shared_ptr<VariableNodeBase>> variables) {
    connectObservation(*this, factor, variables);
}

} // namespace gsolver
