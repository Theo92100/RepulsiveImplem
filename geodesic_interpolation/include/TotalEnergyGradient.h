// TotalEnergyGradient.h
#pragma once

#include <goast/Optimization/optInterface.h>
#include "EnergyCalculator.hpp"
#include <goast/DiscreteShells.h>

template<typename ConfiguratorType>
class TotalEnergyGradient : public BaseOp<typename ConfiguratorType::VectorType, typename ConfiguratorType::VectorType> {
public:
    using VectorType = typename ConfiguratorType::VectorType;
    using ShellDeformationType = ShellDeformation<DefaultConfigurator, NonlinearMembraneDeformation<DefaultConfigurator>, SimpleBendingDeformation<DefaultConfigurator>>;

    TotalEnergyGradient(EnergyCalculator& energy_calculator, const std::vector<Mesh>& initial_path, const ShellDeformationType& W)
        : energy_calculator(energy_calculator), initial_path(initial_path), W(W) {}

    void apply(const VectorType& x, VectorType& gradient) const override {
        // Create a copy of the path
        std::vector<Mesh> path = initial_path;

        // Update the path with the new positions x
        update_path_from_vector(path, x);

        // Compute the total gradient
        Eigen::VectorXd gradient_elastic = energy_calculator.compute_gradient_elastic(path, W);
        Eigen::VectorXd gradient_repulsive = energy_calculator.compute_gradient_repulsive(path);
        gradient = gradient_elastic + gradient_repulsive;
    }

private:
    EnergyCalculator& energy_calculator;
    const std::vector<Mesh> initial_path;
    const ShellDeformationType& W;

      
};
