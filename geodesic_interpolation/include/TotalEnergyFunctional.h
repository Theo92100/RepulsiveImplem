// TotalEnergyFunctional.h
#pragma once

#include <goast/Optimization/optInterface.h>
#include "EnergyCalculator.hpp"
#include <goast/DiscreteShells.h>

template<typename ConfiguratorType>
class TotalEnergyFunctional : public BaseOp<typename ConfiguratorType::VectorType, typename ConfiguratorType::RealType> {
public:
    using VectorType = typename ConfiguratorType::VectorType;
    using RealType = typename ConfiguratorType::RealType;
    using ShellDeformationType = ShellDeformation<DefaultConfigurator, NonlinearMembraneDeformation<DefaultConfigurator>, SimpleBendingDeformation<DefaultConfigurator>>;

    TotalEnergyFunctional(EnergyCalculator& energy_calculator, const std::vector<Mesh>& initial_path, const ShellDeformationType& W)
        : energy_calculator(energy_calculator), initial_path(initial_path), W(W) {}

    void apply(const VectorType& x, RealType& energy) const override {
        // Create a copy of the path
        std::vector<Mesh> path = initial_path;

        // Update the path with the new positions x
        update_path_from_vector(path, x);

        // Compute the total energy
        double elastic_energy = energy_calculator.compute_elastic_energy(path, W);
        double repulsive_energy = energy_calculator.compute_repulsive_energy(path);
        energy = elastic_energy + repulsive_energy;
    }

private:
    EnergyCalculator& energy_calculator;
    const std::vector<Mesh> initial_path;
    const ShellDeformationType& W;

};
