#pragma once

#include <gsolver/graph/factor_graph.h>
#include <gsolver/graph/types/variables/se3_posevel.h>

namespace gsolver {

class RangeFactor final : public FactorNode<1, SE3PoseVel, SE3PoseVel> {
public:
    // Connect the observer first and the target second. Range is in metres.
    RangeFactor(const std::string& id, double range, const MatrixErrorDim& covariance);

    void updateErrorAndJacobian() override;

private:
    double range_;
};

template <>
void FactorGraph::addFactorNode(const std::shared_ptr<RangeFactor>& factor, std::vector<std::string> variable_ids);

template <>
void FactorGraph::addFactorNode(const std::shared_ptr<RangeFactor>& factor,
                                std::vector<std::shared_ptr<VariableNodeBase>> variables);

} // namespace gsolver
