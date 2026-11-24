/******************************************************************************

                                     Turblyze
                           3D incompressible CFD solver
                       Copyright (C) 2025-2026 Mohamed Mousa
                        SPDX-License-Identifier: Apache-2.0

 ------------------------------------------------------------------------------
 * @file SIMPLE.h
 * @brief SIMPLE algorithm for incompressible Navier-Stokes equations
 *
 * @details SIMPLE (Semi-Implicit Method for Pressure-Linked Equations) is a
 * segregated pressure-correction algorithm. It inherits from MomentumTransport
 * and Segregated.
 *****************************************************************************/

#pragma once

// ********************************** Headers *********************************

#include "Segregated.h"

// ******************************* class SIMPLE *******************************

class SIMPLE final : public Segregated
{
public:

// ************************* Special Member Functions *************************

    /// Use the segregated solver constructor
    using Segregated::Segregated;

    /// Copy constructor and assignment - Not copyable (const T& members)
    SIMPLE(const SIMPLE&) = delete;
    SIMPLE& operator=(const SIMPLE&) = delete;

    /// Move constructor and assignment - Not movable (const T& members)
    SIMPLE(SIMPLE&&) = delete;
    SIMPLE& operator=(SIMPLE&&) = delete;

    /// Destructor
    ~SIMPLE() noexcept override = default;

// ****************************** Private Methods *****************************

private:

    /// One SIMPLE outer iteration
    [[nodiscard]] bool outerIteration
    (
        const TransientFields* prevStep
    ) override;

    /// Algorithm label for banners and convergence messages
    [[nodiscard]] Name algorithmName() const noexcept override
    {
        return "SIMPLE";
    }
};
