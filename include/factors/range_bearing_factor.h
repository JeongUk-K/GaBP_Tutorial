#pragma once

#include <gsolver/graph/factor_graph.h>
#include <gsolver/graph/types/variables/se3_posevel.h>

namespace gsolver {

class RangeBearingFactor final : public FactorNode<2, SE3PoseVel, SE3PoseVel> {
public:
    // Connect the observer first and the target second. Measurement is [range (m), azimuth (rad)].
    RangeBearingFactor(const std::string& id, const VectorErrorDim& measurement, const MatrixErrorDim& covariance);

    void updateErrorAndJacobian() override;

private:
    VectorErrorDim measurement_;
};

template <>
void FactorGraph::addFactorNode(const std::shared_ptr<RangeBearingFactor>& factor,
                                std::vector<std::string> variable_ids);

template <>
void FactorGraph::addFactorNode(const std::shared_ptr<RangeBearingFactor>& factor,
                                std::vector<std::shared_ptr<VariableNodeBase>> variables);

} // namespace gsolver
