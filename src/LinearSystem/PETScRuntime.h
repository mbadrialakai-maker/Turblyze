/******************************************************************************

                                     Turblyze
                           3D incompressible CFD solver
                       Copyright (C) 2025-2026 Mohamed Mousa
                        SPDX-License-Identifier: Apache-2.0

 ------------------------------------------------------------------------------
 * @file PETScRuntime.h
 * @brief RAII ownership of the PETSc runtime
 *
 * @details One PETScRuntime is created in main() before any solver object
 * and destroyed after all of them: every PETSc handle (Mat, Vec, KSP) must
 * be created after PetscInitialize and destroyed before PetscFinalize.
 * The MPI runtime is owned by MPIRuntime (constructed first in main()),
 * so PetscInitialize attaches to the already-running MPI as a guest and
 * PetscFinalize leaves it running for MPIRuntime's destructor.
 *****************************************************************************/

#pragma once

// ********************************** Headers *********************************

// Standard library headers
#include <string>
#include <type_traits>

// External library headers
#include <petscsys.h>

// Project headers
#include "Scalar.h"
#include "ErrorHandler.h"

// ************************** Build-Time Invariants ****************************

// Scalar and PETSc's scalar floating-point precision must match
static_assert
(
    std::is_same_v<PetscScalar, Scalar>,
    "PetscScalar must match Scalar: rebuild PETSc with the matching "
    "--with-precision, or flip TURBLYZE_DOUBLE_PRECISION"
);

// *********************** PETSc Version Compatibility ************************

// For Ubuntu users, the apt repos don't carry PETSc 3.22
// In older releases, PETSC_DEFAULT served as both sentinel values.
#ifndef PETSC_CURRENT
#define PETSC_CURRENT PETSC_DEFAULT
#endif

#ifndef PETSC_UNLIMITED
#define PETSC_UNLIMITED PETSC_DEFAULT
#endif

#ifndef PETSC_SUCCESS
#define PETSC_SUCCESS 0
#endif

// ****************************** PETScErrorCheck *****************************

/// Verify a PETSc call has no errors, aborting with a FatalError on failure
#define PETScErrorCheck(call) { PETScAssert((call), #call); }

inline void PETScAssert
(
    PetscErrorCode errorCode,
    const char* call,
    Location location = Location::current()
)
{
    if (errorCode != PETSC_SUCCESS)
    {
        FatalError
        (
            "PETSc error " + std::to_string(errorCode)
          + " in '" + call + "'", location
        );
    }
}

// **************************** class PETScRuntime ****************************

class PETScRuntime
{
public:

// ************************* Special Member Functions *************************

    /// Initialize the PETSc runtime after MPI has been initialized
    PETScRuntime()
    {
        PETScErrorCheck(PetscInitializeNoArguments());
    }

    /// Copy constructor and assignment - Not copyable (owns the runtime)
    PETScRuntime(const PETScRuntime&) = delete;
    PETScRuntime& operator=(const PETScRuntime&) = delete;

    /// Move constructor and assignment - Not movable
    PETScRuntime(PETScRuntime&&) = delete;
    PETScRuntime& operator=(PETScRuntime&&) = delete;

    /// Finalize the PETSc runtime
    ~PETScRuntime() noexcept
    {
        if (PetscFinalize() != PETSC_SUCCESS)
        {
            // Destructors cannot throw: report and continue shutdown
            Warning("PetscFinalize failed");
        }
    }
};
