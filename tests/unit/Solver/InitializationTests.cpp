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

// Standard library headers
#include <cmath>

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
#include "TaylorGreen.h"
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

// ************************ Taylor-Green Initialization ***********************

TEST_CASE
(
    "TaylorGreen initializes analytical 3D vortex velocity and pressure",
    "[initialization]"
)
{
    const TestMesh box(4, 4, 4, S(0.5));
    const Mesh& mesh = box.mesh();
    BoundaryConditions bc;

    ScalarField Ux(mesh);
    ScalarField Uy(mesh);
    ScalarField Uz(mesh);
    ScalarField p(mesh);

    const Scalar U0 = S(2.0);
    const Scalar L = S(0.5);
    const Scalar p0 = S(10.0);
    const Scalar rho = S(1.2);

    const Initialization::TaylorGreen init(U0, L, p0, rho);
    init.initialize(mesh, bc, Ux, Uy, Uz, p);

    const Count numOwned = mesh.numDomainCells();
    const Scalar pFactor = rho * U0 * U0 / S(16.0);

    for (Index i = 0; i < numOwned; ++i)
    {
        const Vector c = mesh.cells()[i].centroid();
        const Scalar x = c.x();
        const Scalar y = c.y();
        const Scalar z = c.z();

        const Scalar expectedUx =
            U0 * std::sin(x / L) * std::cos(y / L) * std::cos(z / L);
        const Scalar expectedUy =
            -U0 * std::cos(x / L) * std::sin(y / L) * std::cos(z / L);
        const Scalar expectedP =
            p0 + pFactor
          * (std::cos(S(2.0) * x / L) + std::cos(S(2.0) * y / L))
          * (std::cos(S(2.0) * z / L) + S(2.0));

        REQUIRE_THAT(Ux[i], WithinAbs(expectedUx, TestTolerances::absTight));
        REQUIRE_THAT(Uy[i], WithinAbs(expectedUy, TestTolerances::absTight));
        REQUIRE_THAT(Uz[i], WithinAbs(S(0.0), TestTolerances::absTight));
        REQUIRE_THAT(p[i], WithinAbs(expectedP, TestTolerances::absTight));
    }
}

