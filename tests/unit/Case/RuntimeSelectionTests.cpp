/******************************************************************************

                                     Turblyze
                           3D incompressible CFD solver
                       Copyright (C) 2025-2026 Mohamed Mousa
                        SPDX-License-Identifier: Apache-2.0

 ------------------------------------------------------------------------------
 * @file RuntimeSelectionTests.cpp
 * @brief Factory positive paths and the advertised name lists
 *****************************************************************************/

// ********************************** Headers *********************************

// Standard library headers
#include <algorithm>

// External library headers
#include <catch2/catch_test_macros.hpp>

// Project headers
#include "GradientScheme.h"
#include "ConvectionScheme.h"
#include "TimeScheme.h"
#include "LinearSolvers.h"
#include "TurbulenceModel.h"
#include "LES.h"
#include "Smagorinsky.h"
#include "WALE.h"
#include "MomentumTransport.h"
#include "BoundaryConditions.h"
#include "Initializer.h"
#include "MeshFixtures.h"
#include "StringTypes.h"

// ***************************** Internal Helpers *****************************

namespace
{

/// Whether a name list carries a given token
[[nodiscard]] bool contains(const NameList& names, const Name& target)
{
    return std::find(names.begin(), names.end(), target) != names.end();
}

} // namespace

// **************************** Gradient Schemes ******************************

TEST_CASE("Gradient-scheme selection", "[selection]")
{
    const NameList names = GradientScheme::availableSchemes();
    REQUIRE(names.size() == Count{1});
    REQUIRE(names.front() == "leastSquares");

    const TestMesh box(2, 1, 1);
    const BoundaryConditions bc;
    REQUIRE(GradientScheme::create("leastSquares", box.mesh(), bc) != nullptr);
}

// **************************** Convection Schemes ****************************

TEST_CASE("Convection-scheme selection", "[selection]")
{
    const NameList names = ConvectionScheme::availableSchemes();
    REQUIRE(contains(names, "Upwind"));
    REQUIRE(contains(names, "CentralDifference"));
    REQUIRE(contains(names, "SecondOrderUpwind"));
    REQUIRE(contains(names, "LUST"));

    for (const Name& name : names)
    {
        REQUIRE(ConvectionScheme::create(name) != nullptr);
    }
}

// ******************************* Time Schemes *******************************

TEST_CASE("Time-scheme selection", "[selection]")
{
    const NameList names = TimeScheme::availableSchemes();
    REQUIRE(contains(names, "steadyState"));
    REQUIRE(contains(names, "implicitEuler"));
    REQUIRE(contains(names, "CrankNicolson"));
    REQUIRE(contains(names, "secondOrderImplicit"));

    for (const Name& name : names)
    {
        REQUIRE(TimeScheme::create(name) != nullptr);
    }
}

// ***************************** Linear Solvers ******************************

TEST_CASE("Linear-solver selection lists every supported method", "[selection]")
{
    const NameList solvers = LinearSolver::availableSolvers();
    REQUIRE(contains(solvers, "BiCGSTAB"));
    REQUIRE(contains(solvers, "PCG"));
    REQUIRE(contains(solvers, "GMRES"));
    REQUIRE(contains(solvers, "FGMRES"));
    REQUIRE(contains(solvers, "TFQMR"));
    REQUIRE(contains(solvers, "CGS"));
    REQUIRE(contains(solvers, "MINRES"));
    REQUIRE(contains(solvers, "Richardson"));
    REQUIRE(contains(solvers, "Chebyshev"));
    REQUIRE(contains(solvers, "PreOnly"));

    const NameList pcs = LinearSolver::availablePreconditioners();
    REQUIRE(contains(pcs, "Jacobi"));
    REQUIRE(contains(pcs, "None"));
    REQUIRE(contains(pcs, "ILU"));
    REQUIRE(contains(pcs, "ICC"));
    REQUIRE(contains(pcs, "SOR"));
    REQUIRE(contains(pcs, "GaussSeidel"));
    REQUIRE(contains(pcs, "AMG"));
    REQUIRE(contains(pcs, "GAMG"));
    REQUIRE(contains(pcs, "BlockJacobi"));
    REQUIRE(contains(pcs, "ASM"));
    REQUIRE(contains(pcs, "LU"));
    REQUIRE(contains(pcs, "Cholesky"));
}

// **************************** Turbulence Models ****************************

TEST_CASE("Turbulence-model selection", "[selection]")
{
    const NameList names = TurbulenceModel::availableModels();
    REQUIRE(contains(names, "Laminar"));
    REQUIRE(contains(names, "kOmegaSST"));
    REQUIRE(contains(names, "Smagorinsky"));
    REQUIRE(contains(names, "WALE"));

    REQUIRE(TurbulenceModel::isLaminar("Laminar"));
    REQUIRE_FALSE(TurbulenceModel::isLaminar("kOmegaSST"));
    REQUIRE_FALSE(TurbulenceModel::isLaminar("Smagorinsky"));
    REQUIRE_FALSE(TurbulenceModel::isLaminar("WALE"));

    REQUIRE(TurbulenceModel::isRANS("kOmegaSST"));
    REQUIRE_FALSE(TurbulenceModel::isRANS("Smagorinsky"));
    REQUIRE_FALSE(TurbulenceModel::isRANS("WALE"));
    REQUIRE_FALSE(TurbulenceModel::isRANS("Laminar"));

    REQUIRE(TurbulenceModel::isLES("Smagorinsky"));
    REQUIRE(TurbulenceModel::isLES("WALE"));
    REQUIRE_FALSE(TurbulenceModel::isLES("kOmegaSST"));
    REQUIRE_FALSE(TurbulenceModel::isLES("Laminar"));

    const NameList lesNames = LES::availableModels();
    REQUIRE(contains(lesNames, "Smagorinsky"));
    REQUIRE(contains(lesNames, "WALE"));

    const TestMesh box(2, 1, 1);
    const BoundaryConditions bc;
    const auto lesModel =
        LES::create
        (
            "Smagorinsky",
            box.mesh(),
            bc,
            S(1e-4),
            Smagorinsky::coeffs_.Cs,
            Smagorinsky::coeffs_.Ck,
            WALE::coeffs_.Cw,
            true,
            Smagorinsky::coeffs_.Aplus,
            Smagorinsky::coeffs_.kappa,
            false
        );
    REQUIRE(lesModel != nullptr);
    REQUIRE(lesModel->isTurbulent() == true);
    REQUIRE(lesModel->wallDistanceConverged() == true);
    REQUIRE(dynamic_cast<const Smagorinsky*>(lesModel.get()) != nullptr);

    const auto waleModel =
        LES::create
        (
            "WALE",
            box.mesh(),
            bc,
            S(1e-4),
            Smagorinsky::coeffs_.Cs,
            WALE::coeffs_.Ck,
            WALE::coeffs_.Cw,
            true,
            Smagorinsky::coeffs_.Aplus,
            Smagorinsky::coeffs_.kappa,
            false
        );
    REQUIRE(waleModel != nullptr);
    REQUIRE(waleModel->isTurbulent() == true);
    REQUIRE(waleModel->wallDistanceConverged() == true);
    REQUIRE(dynamic_cast<const WALE*>(waleModel.get()) != nullptr);
}

// *********************** MomentumTransport Algorithms ***********************

TEST_CASE("MomentumTransport algorithm selection", "[selection]")
{
    const NameList names = MomentumTransport::availableAlgorithms();
    REQUIRE(contains(names, "SIMPLE"));
    REQUIRE(contains(names, "PISO"));
}

// **************************** Flow Initialization ***************************

TEST_CASE("Flow-initialization selection", "[selection]")
{
    const NameList names = Initializer::availableTypes();
    REQUIRE(names.size() == Count{3});
    REQUIRE(contains(names, "Uniform"));
    REQUIRE(contains(names, "taylorGreen"));
    REQUIRE(contains(names, "potentialFlow"));
}
