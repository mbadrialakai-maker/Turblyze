/******************************************************************************

                                     Turblyze
                           3D incompressible CFD solver
                       Copyright (C) 2025-2026 Mohamed Mousa
                        SPDX-License-Identifier: Apache-2.0

 ------------------------------------------------------------------------------
 * @file LESModelTests.cpp
 * @brief Unit tests for LES base class, Smagorinsky, and WALE SGS models
 *****************************************************************************/

// ********************************** Headers *********************************

// Standard library headers
#include <cmath>

// External library headers
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

// Project headers
#include "MeshFixtures.h"
#include "TestTolerances.h"
#include "LES.h"
#include "Smagorinsky.h"
#include "WALE.h"
#include "BoundaryConditions.h"
#include "NoSlip.h"
#include "Tensor.h"
#include "CellData.h"

using Catch::Matchers::WithinAbs;
using Catch::Matchers::WithinRel;

// ******************************* Filter Width *******************************

TEST_CASE("LES filter width calculation", "[models][LES]")
{
    // 2x1x1 hex box with dx = 2.0: 2 cube cells with V = 8.0 each
    const TestMesh box(2, 1, 1, S(2.0));
    const BoundaryConditions bc;
    const Smagorinsky model
    (
        box.mesh(),
        bc,
        S(1e-4),
        S(0.165),
        S(0.094),
        false // no damping
    );

    const auto& delta = model.filterWidth();
    REQUIRE(delta.size() >= 2);

    // Delta = V^(1/3) = 8.0^(1/3) = 2.0
    REQUIRE_THAT
    (
        delta[0],
        WithinRel(S(2.0), TestTolerances::relTight)
    );
    REQUIRE_THAT
    (
        delta[1],
        WithinRel(S(2.0), TestTolerances::relTight)
    );
}

// *************************** Smagorinsky Viscosity **************************

TEST_CASE
(
    "Smagorinsky sub-grid viscosity and TKE without damping",
    "[models][LES]"
)
{
    const TestMesh box(2, 1, 1);
    const BoundaryConditions bc;
    const Scalar Cs = S(0.1);
    const Scalar Ck = S(0.094);
    const Scalar nu = S(1e-3);

    Smagorinsky model(box.mesh(), bc, nu, Cs, Ck, false);

    // Pure shear: dU/dy = 2.0 -> gradU = [[0, 2, 0], [0, 0, 0], [0, 0, 0]]
    // Symmetric strain S = [[0, 1, 0], [1, 0, 0], [0, 0, 0]]
    // |S| = sqrt(2 * (1^2 + 1^2)) = 2.0
    TensorField gradU(box.mesh());
    const Tensor pureShear
    (
        S(0.0), S(2.0), S(0.0),
        S(0.0), S(0.0), S(0.0),
        S(0.0), S(0.0), S(0.0)
    );
    gradU.setAll(pureShear);

    ScalarField Ux(box.mesh(), S(0.0));
    ScalarField Uy(box.mesh(), S(0.0));
    ScalarField Uz(box.mesh(), S(0.0));
    FaceFluxField flowRate(box.mesh(), S(0.0));

    model.solve(Ux, Uy, Uz, flowRate, gradU);

    // Filter width delta = 1.0
    // ls = Cs * delta = 0.1 * 1.0 = 0.1
    // nut = ls^2 * |S| = 0.01 * 2.0 = 0.02
    // k_sgs = Ck * delta^2 * |S|^2 = 0.094 * 1.0 * 4.0 = 0.376
    const Scalar expectedNut = S(0.02);
    const Scalar expectedKSgs = S(0.376);

    const auto& nut = model.turbulentViscosity();
    const auto& kSgs = model.kSgs();

    REQUIRE_THAT
    (
        nut[0],
        WithinRel(expectedNut, TestTolerances::relTight)
    );
    REQUIRE_THAT
    (
        kSgs[0],
        WithinRel(expectedKSgs, TestTolerances::relTight)
    );

    // Cell data outputs registration
    const auto outputs = model.cellDataOutputs();
    REQUIRE(outputs.size() == 3);
    REQUIRE(outputs[0].first == "nut");
    REQUIRE(outputs[1].first == "k_sgs");
    REQUIRE(outputs[2].first == "delta");
}

// ************************** Van Driest Wall Damping *************************

TEST_CASE
(
    "Smagorinsky with Van Driest damping reduces near-wall viscosity",
    "[models][LES]"
)
{
    const TestMesh box(1, 2, 1);

    BoundaryConditions::BCs bcs;
    bcs[BoxPatch::yMin][Field::Ux] = std::make_unique<NoSlip>();
    bcs[BoxPatch::yMin][Field::Uy] = std::make_unique<NoSlip>();
    bcs[BoxPatch::yMin][Field::Uz] = std::make_unique<NoSlip>();
    const BoundaryConditions bc(std::move(bcs), box.mesh());

    const Scalar Cs = S(0.165);
    const Scalar Ck = S(0.094);
    const Scalar nu = S(0.1);
    const Scalar Aplus = Smagorinsky::coeffs_.Aplus;
    const Scalar kappa = Smagorinsky::coeffs_.kappa;

    Smagorinsky modelDamped(box.mesh(), bc, nu, Cs, Ck, true, Aplus, kappa);
    Smagorinsky modelUndamped(box.mesh(), bc, nu, Cs, Ck, false, Aplus, kappa);

    TensorField gradU(box.mesh());
    const Tensor shear
    (
        S(0.0), S(10.0), S(0.0),
        S(0.0), S(0.0),  S(0.0),
        S(0.0), S(0.0),  S(0.0)
    );
    gradU.setAll(shear);

    ScalarField Ux(box.mesh(), S(0.0));
    ScalarField Uy(box.mesh(), S(0.0));
    ScalarField Uz(box.mesh(), S(0.0));
    FaceFluxField flowRate(box.mesh(), S(0.0));

    modelDamped.solve(Ux, Uy, Uz, flowRate, gradU);
    modelUndamped.solve(Ux, Uy, Uz, flowRate, gradU);

    // Cell 0 is at y = 0.5 from yMin wall; |S| = 10, u_tau = sqrt(0.1 * 10) = 1
    // y+ = 0.5 * 1 / 0.1 = 5.0; ls_wall = kappa * y * (1 - exp(-y+/A+)) < Cs * d
    const Scalar y0 = S(0.5);
    const Scalar magS = S(10.0);
    const Scalar yPlus0 = y0 * std::sqrt(nu * magS) / nu;
    const Scalar lsWall0 = kappa * y0 * (S(1.0) - std::exp(-yPlus0 / Aplus));
    const Scalar expectedNut0 = lsWall0 * lsWall0 * magS;

    REQUIRE_THAT
    (
        modelDamped.turbulentViscosity()[0],
        WithinRel(expectedNut0, TestTolerances::relTight)
    );
    REQUIRE
    (
        modelDamped.turbulentViscosity()[0]
      < modelUndamped.turbulentViscosity()[0]
    );
    // Near-wall cell (y = 0.5) is damped more strongly than outer cell (y = 1.5)
    REQUIRE
    (
        modelDamped.turbulentViscosity()[0]
      < modelDamped.turbulentViscosity()[1]
    );
}

// ****************************** WALE Viscosity ******************************

TEST_CASE
(
    "WALE sub-grid viscosity vanishes in pure shear and zero gradient",
    "[models][les]"
)
{
    const TestMesh box(2, 1, 1);
    const BoundaryConditions bc;
    WALE model(box.mesh(), bc, S(1e-4));

    REQUIRE_THAT
    (
        model.Cw(),
        WithinRel(WALE::coeffs_.Cw, TestTolerances::relTight)
    );
    REQUIRE_THAT
    (
        model.Ck(),
        WithinRel(WALE::coeffs_.Ck, TestTolerances::relTight)
    );

    ScalarField Ux(box.mesh(), S(0.0));
    ScalarField Uy(box.mesh(), S(0.0));
    ScalarField Uz(box.mesh(), S(0.0));
    FaceFluxField flowRate(box.mesh(), S(0.0));

    // Zero velocity gradient -> nut = 0, k_sgs = 0 (finite, no NaN)
    TensorField gradU(box.mesh());
    gradU.setAll(Tensor{});
    model.solve(Ux, Uy, Uz, flowRate, gradU);

    REQUIRE(std::isfinite(model.turbulentViscosity()[0]));
    REQUIRE(std::isfinite(model.kSgs()[0]));
    REQUIRE_THAT
    (
        model.turbulentViscosity()[0],
        WithinAbs(S(0.0), TestTolerances::absTight)
    );
    REQUIRE_THAT
    (
        model.kSgs()[0],
        WithinAbs(S(0.0), TestTolerances::absTight)
    );

    // Pure shear: gradU = [[0, 2, 0], [0, 0, 0], [0, 0, 0]] -> gradU^2 = 0
    // so Sd = 0 and WALE produces exact zero SGS viscosity
    const Tensor pureShear
    (
        S(0.0), S(2.0), S(0.0),
        S(0.0), S(0.0), S(0.0),
        S(0.0), S(0.0), S(0.0)
    );
    gradU.setAll(pureShear);
    model.solve(Ux, Uy, Uz, flowRate, gradU);

    REQUIRE_THAT
    (
        model.turbulentViscosity()[0],
        WithinAbs(S(0.0), TestTolerances::absTight)
    );
    REQUIRE_THAT
    (
        model.kSgs()[0],
        WithinAbs(S(0.0), TestTolerances::absTight)
    );
}

TEST_CASE
(
    "WALE sub-grid viscosity and TKE for plane strain and solid-body rotation",
    "[models][LES]"
)
{
    const TestMesh box(2, 1, 1);
    const BoundaryConditions bc;
    const Scalar Cw = S(0.325);
    const Scalar Ck = S(0.094);
    WALE model(box.mesh(), bc, S(1e-4), Cw, Ck);

    ScalarField Ux(box.mesh(), S(0.0));
    ScalarField Uy(box.mesh(), S(0.0));
    ScalarField Uz(box.mesh(), S(0.0));
    FaceFluxField flowRate(box.mesh(), S(0.0));
    TensorField gradU(box.mesh());

    // Solid-body rotation: gradU = [[0, -w, 0], [w, 0, 0], [0, 0, 0]]
    // S = 0 -> S:S = 0
    // gradU^2 = diag(-w^2, -w^2, 0), trace = -2 w^2
    // Sd = diag(-w^2/3, -w^2/3, 2 w^2/3) -> Sd:Sd = (2/3) w^4
    // nut = (Cw * delta)^2 * (Sd:Sd)^(3/2) / (Sd:Sd)^(5/4)
    //     = (Cw * delta)^2 * (Sd:Sd)^(1/4) = Cw^2 * w * (2/3)^(1/4)
    const Scalar w = S(3.0);
    const Tensor rotation
    (
        S(0.0), -w,     S(0.0),
        w,      S(0.0), S(0.0),
        S(0.0), S(0.0), S(0.0)
    );
    gradU.setAll(rotation);
    model.solve(Ux, Uy, Uz, flowRate, gradU);

    const Scalar expectedNutRot =
        Cw * Cw * w * std::pow(S(2.0) / S(3.0), S(0.25));
    REQUIRE_THAT
    (
        model.turbulentViscosity()[0],
        WithinRel(expectedNutRot, TestTolerances::relTight)
    );

    // Plane strain: gradU = diag(1, -1, 0)
    // S = diag(1, -1, 0) -> S:S = 2
    // gradU^2 = diag(1, 1, 0), trace = 2
    // Sd = diag(1/3, 1/3, -2/3) -> Sd:Sd = 2/3
    const Tensor planeStrain
    (
        S(1.0), S(0.0),  S(0.0),
        S(0.0), S(-1.0), S(0.0),
        S(0.0), S(0.0),  S(0.0)
    );
    gradU.setAll(planeStrain);
    model.solve(Ux, Uy, Uz, flowRate, gradU);

    const Scalar sdSqr = S(2.0) / S(3.0);
    const Scalar expectedNutStrain =
        Cw * Cw * std::pow(sdSqr, S(1.5))
      / (std::pow(S(2.0), S(2.5)) + std::pow(sdSqr, S(1.25)));
    const Scalar expectedKSgsStrain =
        (expectedNutStrain / Ck) * (expectedNutStrain / Ck);

    REQUIRE_THAT
    (
        model.turbulentViscosity()[0],
        WithinRel(expectedNutStrain, TestTolerances::relTight)
    );
    REQUIRE_THAT
    (
        model.kSgs()[0],
        WithinRel(expectedKSgsStrain, TestTolerances::relTight)
    );

    // Cell data outputs registration
    const auto outputs = model.cellDataOutputs();
    REQUIRE(outputs.size() == 3);
    REQUIRE(outputs[0].first == "nut");
    REQUIRE(outputs[1].first == "k_sgs");
    REQUIRE(outputs[2].first == "delta");
}
