# DDMFaultSlip

*A package for modelling fault and fracture mechanics using the Displacement Discontinuity Method (DDM)*

[![License](https://img.shields.io/github/license/mashape/apistatus.svg?maxAge=2592000)](https://github.com/ajacquey/DDMFaultSlip.jl/blob/main/LICENSE)
[![Build Status](https://github.com/ajacquey/DDMFaultSlip.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/ajacquey/DDMFaultSlip.jl/actions/workflows/CI.yml?query=branch%3Amain)
[![Coverage](https://codecov.io/gh/ajacquey/DDMFaultSlip.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/ajacquey/DDMFaultSlip.jl)
[![](https://img.shields.io/badge/docs-stable-blue.svg)](https://ajacquey.github.io/DDMFaultSlip.jl/stable/)
[![](https://img.shields.io/badge/docs-dev-blue.svg)](https://ajacquey.github.io/DDMFaultSlip.jl/dev/)

`DDMFaultSlip.jl` simulates the quasi-static mechanics of faults and fractures in an
infinite, homogeneous, linear elastic medium. Only the fault or fracture surface is
discretized, and the unknowns are the displacement discontinuities (opening and
slip) across it. The package was designed to model **fluid-induced aseismic slip**,
the slow frictional slip of faults caused by pore pressure changes, for example
during fluid injection.

## Features

- **Geometries**: 2D plane strain faults, 3D axisymmetric faults and 3D planar
  faults of arbitrary shape (triangular meshes generated with [Gmsh](https://gmsh.info)).
- **Problems**: opening (mode I) and shear (modes II/III) displacement discontinuities.
- **Interface laws**: constant friction, linear slip-weakening friction, constant
  yield strength, Dugdale cohesive zone, and user-defined constraints.
- **Fluid coupling**: pore pressure prescribed as a function of space and time.
- **Efficient solvers**: hierarchical matrix compression of the elastic matrix and a
  Newton–Krylov solver with a hierarchical LU preconditioner, with multithreading.
- **Outputs**: CSV files (2D problems) and VTK files (3D problems, for ParaView).

## Installation

You can install `DDMFaultSlip` directly from the `Pkg` REPL (press `]` in the Julia REPL):

```
pkg> add DDMFaultSlip
```

Start Julia with several threads (e.g. `julia -t auto`) to speed up simulations.

## Quick start

Opening of a 2D crack under a uniform tensile stress (stresses are positive in
compression):

```julia
using DDMFaultSlip
using StaticArrays

mesh = DDMesh1D(SVector(-1.0, 0.0), SVector(1.0, 0.0), 200)
problem = NormalDDProblem(mesh; μ=1.0)
addConstraint!(problem, FunctionConstraint((X, Δw) -> -1.0))
run!(problem)

problem.w.value # opening of the elements, ≈ sqrt(1 - x^2)
```

## Example: fluid-induced aseismic slip

A 2D fault with constant friction slips in response to a fluid injected at
constant pressure at `x = 0`:

```julia
using DDMFaultSlip
using StaticArrays
using SpecialFunctions

μ, f, h = 1.0, 0.5, 1.0e-05 # shear modulus, friction coefficient, fault thickness
σ₀, τ₀ = 1.0, 0.375         # initial normal and shear stresses
Δp, α = 0.5, 1.0            # injection pressure and hydraulic diffusivity

mesh = DDMesh1D(SVector(-1.2, 0.0), SVector(1.2, 0.0), 400)
problem = ShearDDProblem(mesh; μ=μ)

addNormalStressIC!(problem, X -> σ₀ * ones(length(X)))
addShearStressIC!(problem, X -> τ₀ * ones(length(X)))
addFluidCoupling!(problem, FunctionPressure(mesh, (X, t) -> Δp * erfc.([abs(x[1]) for x in X] / sqrt(α * t))))
addFrictionConstraint!(problem, ConstantFriction(f, μ / h))

run!(problem, TimeSequence(collect(range(0.0, 1.0, length=21))))

problem.δ.value # slip of the elements at t = 1
```

The slipping region grows as `λ√(αt)`, in agreement with the analytical solution of
[Bhattacharya & Viesca (2019)](https://doi.org/10.1126/science.aaw7354). See the
[tutorial](https://ajacquey.github.io/DDMFaultSlip.jl/dev/tutorials/fluid_injection/)
for the full example and its comparison with the analytical solution. More examples,
including 3D simulations, are in the [`examples`](examples) folder.

## Documentation

The [documentation](https://ajacquey.github.io/DDMFaultSlip.jl/stable/) contains
the formulation, a user guide, tutorials and the API reference.

## Citing

If you use `DDMFaultSlip.jl` in your work, please cite it using the
[`CITATION.bib`](CITATION.bib) file.

## Related packages

`DDMFaultSlip` relies on the following essential packages:
* [HMatrices.jl](https://github.com/IntegralEquations/HMatrices.jl): used for hierarchical matrices handling.
* [IterativeSolvers.jl](https://github.com/JuliaLinearAlgebra/IterativeSolvers.jl): used for solving linear systems.
