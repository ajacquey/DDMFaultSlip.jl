```@meta
CurrentModule = DDMFaultSlip
```

# [DDMFaultSlip](@id home-section)

*A package for modelling fault and fracture mechanics using the Displacement Discontinuity Method (DDM)*

## Overview

`DDMFaultSlip.jl` is a Julia package for simulating the quasi-static mechanics of
faults and fractures embedded in an infinite, homogeneous, linear elastic medium.
It uses the Displacement Discontinuity Method, a boundary element method in which
only the fault or fracture surface is discretized, and the unknowns are the
displacement discontinuities (opening and slip) across it.

The package was designed to model **fluid-induced aseismic slip**: the slow,
stable frictional slip of a fault caused by pore pressure changes, for example
during fluid injection. It also handles general opening and shear crack problems.

## Features

- **Geometries**: 2D plane strain faults (1D meshes), 3D axisymmetric faults (1D
  radial meshes) and 3D planar faults of arbitrary shape (2D triangular meshes
  generated with [Gmsh](https://gmsh.info)).
- **Problems**: opening (mode I) and shear (modes II/III) displacement discontinuities.
- **Interface laws**: constant friction, linear slip-weakening friction, constant
  yield strength, Dugdale cohesive zone, and user-defined constraints.
- **Fluid coupling**: pore pressure prescribed as a function of space and time,
  acting on the effective normal stress.
- **Efficient solvers**: the dense elastic matrix is compressed as a hierarchical
  matrix ([HMatrices.jl](https://github.com/IntegralEquations/HMatrices.jl)), and the
  nonlinear problem is solved with a Newton–Krylov method preconditioned by a
  hierarchical LU factorization. Assembly and matrix-vector products are multithreaded.
- **Outputs**: CSV files for 2D problems and VTK files for 3D problems (ParaView).

## Installation

`DDMFaultSlip` can be installed from the Julia `Pkg` REPL (press `]` in the Julia REPL):

```
pkg> add DDMFaultSlip
```

Most simulations benefit from multithreading: start Julia with several threads,
e.g. `julia -t auto`.

## Quick start

The following example computes the opening of a 2D crack of half-length 1 under a
uniform tensile stress (stresses are positive in compression) and compares it to
the analytical solution ``w(x) = \sqrt{1 - x^2} / μ``.

```@example quickstart
using DDMFaultSlip
using StaticArrays

# Mesh: 200 elements between x = -1 and x = 1
mesh = DDMesh1D(SVector(-1.0, 0.0), SVector(1.0, 0.0), 200)

# Opening problem with shear modulus μ = 1
problem = NormalDDProblem(mesh; μ=1.0)

# Uniform tensile stress applied on the crack faces
addConstraint!(problem, FunctionConstraint((X, Δw) -> -1.0))

# Solve
run!(problem; log=false)

# Compare with the analytical solution
x = [e.X[1] for e in mesh.elems]
w_exact = sqrt.(1.0 .- x .^ 2)
maximum(abs.(problem.w.value .- w_exact)) / maximum(w_exact)
```

The largest relative error, about 2.5%, is located at the crack tips, where the
opening varies the fastest; it decreases as the mesh is refined.

See the [tutorial](@ref tutorial-injection) for a complete example of fluid-induced
aseismic slip, and the [user guide](@ref guide-section) for an overview of the
package's components.

## Citing

If you use `DDMFaultSlip.jl` in your work, please cite it using the
[`CITATION.bib`](https://github.com/ajacquey/DDMFaultSlip.jl/blob/main/CITATION.bib)
file of the repository.

## Related packages

`DDMFaultSlip` relies on the following essential packages:
* [HMatrices.jl](https://github.com/IntegralEquations/HMatrices.jl): hierarchical matrices.
* [IterativeSolvers.jl](https://github.com/JuliaLinearAlgebra/IterativeSolvers.jl): iterative linear solvers.
