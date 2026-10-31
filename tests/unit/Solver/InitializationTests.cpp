/******************************************************************************

                                     Turblyze
                           3D incompressible CFD solver
                       Copyright (C) 2025-2026 Mohamed Mousa
                        SPDX-License-Identifier: Apache-2.0

 ------------------------------------------------------------------------------
 * @file InitializationTests.cpp
 * @brief Unit tests for flow field initialization methods
 *****************************************************************************/

// ********************************** Headers *********************************

// External library headers
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

// Project headers
#include "MeshFixtures.h"
#include "BoundaryConditions.h"
#include "CellData.h"
#include "Vector.h"
#include "Scalar.h"
#include "Uniform.h"
#include "TestTolerances.h"

using Catch::Matchers::WithinAbs;

// ************************** Uniform Initialization **************************

TEST_CASE
(
    "Uniform initializes uniform velocity and pressure",
    "[initialization]"
)
{
    const TestMesh box(3, 3, 3);
    const Mesh& mesh = box.mesh();
    BoundaryConditions bc;

    ScalarField Ux(mesh);
    ScalarField Uy(mesh);
    ScalarField Uz(mesh);
    ScalarField p(mesh);

    const Vector targetU(S(1.5), S(-2.0), S(3.5));
    const Scalar targetP = S(101325.0);

    const Initialization::Uniform init(targetU, targetP);
    init.initialize(mesh, bc, Ux, Uy, Uz, p);

    const Count numOwned = mesh.numDomainCells();
    REQUIRE(numOwned == Count{27});

    for (Index i = 0; i < numOwned; ++i)
    {
        REQUIRE_THAT(Ux[i], WithinAbs(targetU.x(), TestTolerances::absTight));
        REQUIRE_THAT(Uy[i], WithinAbs(targetU.y(), TestTolerances::absTight));
        REQUIRE_THAT(Uz[i], WithinAbs(targetU.z(), TestTolerances::absTight));
        REQUIRE_THAT(p[i], WithinAbs(targetP, TestTolerances::absTight));
    }
}

TEST_CASE
(
    "Uniform initializes zero velocity quiescent state",
    "[initialization]"
)
{
    const TestMesh box(2, 2, 2);
    const Mesh& mesh = box.mesh();
    BoundaryConditions bc;

    ScalarField Ux(mesh);
    ScalarField Uy(mesh);
    ScalarField Uz(mesh);
    ScalarField p(mesh);

    const Vector zeroU(S(0.0), S(0.0), S(0.0));
    const Scalar refP = S(0.0);

    const Initialization::Uniform init(zeroU, refP);
    init.initialize(mesh, bc, Ux, Uy, Uz, p);

    const Count numOwned = mesh.numDomainCells();
    for (Index i = 0; i < numOwned; ++i)
    {
        REQUIRE_THAT(Ux[i], WithinAbs(S(0.0), TestTolerances::absTight));
        REQUIRE_THAT(Uy[i], WithinAbs(S(0.0), TestTolerances::absTight));
        REQUIRE_THAT(Uz[i], WithinAbs(S(0.0), TestTolerances::absTight));
        REQUIRE_THAT(p[i], WithinAbs(S(0.0), TestTolerances::absTight));
    }
}

