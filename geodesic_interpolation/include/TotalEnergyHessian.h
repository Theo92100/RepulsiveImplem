// TotalEnergyHessian.h
#pragma once

#include <goast/Optimization/optInterface.h>
#include "EnergyCalculator.hpp"
#include <goast/Core.h>
#include <goast/DiscreteShells.h>

template<typename ConfiguratorType>
class TotalEnergyHessian : public BaseOp<typename ConfiguratorType::VectorType, typename ConfiguratorType::SparseMatrixType> {
public:
    using VectorType = typename ConfiguratorType::VectorType;
    using SparseMatrixType = typename ConfiguratorType::SparseMatrixType;
    using ShellDeformationType = ShellDeformation<DefaultConfigurator, NonlinearMembraneDeformation<DefaultConfigurator>, SimpleBendingDeformation<DefaultConfigurator>>;

    TotalEnergyHessian(EnergyCalculator& energy_calculator, const std::vector<Mesh>& initial_path, const ShellDeformationType& W)
        : energy_calculator(energy_calculator), initial_path(initial_path), W(W) {}

    void apply(const VectorType& x, SparseMatrixType& hessian) const override {
        // Create a copy of the path
        std::vector<Mesh> path = initial_path;

        // Update the path with the new positions x
        update_path_from_vector(path, x);

        // Compute the total Hessian
        Eigen::SparseMatrix<double> hessian_elastic = energy_calculator.compute_hessian_elastic(path, W);
        Eigen::SparseMatrix<double> H_GN = construct_HGN(path, energy_calculator);
        hessian = hessian_elastic + H_GN;
    }

private:
    EnergyCalculator& energy_calculator;
    const std::vector<Mesh> initial_path;
    const ShellDeformationType& W;

};
