# [User guide](@id guide-section)

A simulation is set up in a few steps:

1. create a mesh of the fault or fracture,
2. create a problem on this mesh,
3. set the initial conditions, constraints and fluid coupling,
4. choose a time stepper and outputs,
5. run the problem with [`run!`](@ref).

## Meshes

### 2D and axisymmetric problems

A [`DDMesh1D`](@ref) is a straight line of uniform elements between two points:

```julia
using DDMFaultSlip
using StaticArrays

mesh = DDMesh1D(SVector(-1.0, 0.0), SVector(1.0, 0.0), 200)
```

For axisymmetric problems (`axisymmetric=true`), the mesh represents the radial
profile of a circular fault, from its center `SVector(0.0, 0.0)` to its outer radius.

### 3D problems

A [`DDMesh2D`](@ref) is a triangular mesh of a planar fault in the ``z = 0`` plane,
read from a [Gmsh](https://gmsh.info) file (format 4.1). For instance, the
following Gmsh script `mesh.geo` creates a circular fault of radius `R`, refined
around its center:

```
lc = 0.02*R;

Point(1) = {0.0, 0.0, 0.0, lc};
Point(2) = {R, 0.0, 0.0, lc};
Point(3) = {0.0, R, 0.0, lc};
Point(4) = {-R, 0.0, 0.0, lc};
Point(5) = {0.0, -R, 0.0, lc};

Circle(1) = {2, 1, 3};
Circle(2) = {3, 1, 4};
Circle(3) = {4, 1, 5};
Circle(4) = {5, 1, 2};

Line Loop(1) = {1, 2, 3, 4};
Plane Surface(1) = {1};
Physical Surface(1) = {1};
```

The mesh can be generated and read from a Julia script:

```julia
R = 2.0
run(`gmsh -2 $(@__DIR__)/mesh.geo -setnumber R $(R) -o $(@__DIR__)/mesh.msh`)
mesh = DDMesh2D(Float64, "mesh.msh")
```

!!! note
    The mesh file name given to `DDMesh2D` is relative to the directory of the
    script calling it.

## Problems

Two types of problems are available:

- [`NormalDDProblem`](@ref): the unknown is the opening `w`.
- [`ShearDDProblem`](@ref): the unknown is the slip `δ`.

Both take the elastic shear modulus `μ` and Poisson's ratio `ν` as keyword arguments:

```julia
problem = ShearDDProblem(mesh; μ=1.0)
```

After a simulation, the current values of the fields at the element centroids are
stored in the problem:

| Field | Description |
|:------|:------------|
| `problem.w.value` | opening ([`NormalDDProblem`](@ref)) |
| `problem.δ.value` | slip ([`ShearDDProblem`](@ref)) |
| `problem.σ.value` | effective normal stress |
| `problem.τ.value` | shear stress ([`ShearDDProblem`](@ref)) |
| `problem.fluid_coupling.p` | pore pressure (with fluid coupling) |

The values at the previous time step are stored in the corresponding `value_old` fields.

## Initial conditions

Initial conditions are given as functions of the vector `X` of element centroids,
returning a vector of values:

```julia
σ₀ = 1.0
τ₀ = 0.4
addNormalStressIC!(problem, X -> σ₀ * ones(length(X)))
addShearStressIC!(problem, X -> τ₀ * ones(length(X)))
```

The initial DD can be set with [`addNormalDDIC!`](@ref) and [`addShearDDIC!`](@ref).
Unspecified initial conditions are zero.

## Constraints

Constraints define the behaviour of the fault or fracture (see the
[background](@ref background-section) section):

```julia
# Coulomb friction with friction coefficient f, fault thickness h
addFrictionConstraint!(problem, ConstantFriction(f, μ / h))

# Linear slip-weakening friction
addFrictionConstraint!(problem, SlipWeakeningFriction(fₚ, fᵣ, δᵣ, μ / h))

# Dugdale cohesive zone (opening problems)
addCohesiveZoneConstraint!(problem, DugdaleCohesiveZone(σₚ, σᵣ, l, μ / h))

# User-defined stress change as a function of position and DD increment
addConstraint!(problem, FunctionConstraint((X, Δu) -> -1.0))
```

## Fluid coupling

The pore pressure is prescribed by a function of the element centroids `X` and of
time, returning the vector of pressures. For instance, for a constant-pressure
injection on a line in 2D:

```julia
using SpecialFunctions

Δp = 0.5 # pressure increase at the injection point
α = 1.0  # hydraulic diffusivity
pressure(X, t) = Δp * erfc.([abs(x[1]) for x in X] / sqrt(α * t))

addFluidCoupling!(problem, FunctionPressure(mesh, pressure))
```

## Time stepping

Transient simulations require a time stepper:

```julia
# 100 time steps of constant size between t = 0 and t = 1
time_stepper = ConstantDT(0.0, 1.0, 100)

# A given sequence of times, here logarithmically spaced
time_stepper = TimeSequence(exp10.(range(-3, 0, length=100)); start_time=0.0)
```

A simulation without time stepper (`run!(problem)`) solves a single load step.

## Outputs

Outputs are written after each time step:

```julia
outputs = [
    CSVDomainOutput(mesh, "outputs/fault"),   # 2D: one CSV file per time step
    CSVMaximumOutput("outputs/fault_max"),    # maximum values over time
]
# For 3D problems: VTKDomainOutput(mesh, "outputs/fault") (ParaView files)

run!(problem, time_stepper; outputs=outputs, output_initial=true)
```

Relative file names are resolved from the directory of the calling script. When
working in the REPL, use absolute file names instead (e.g. `joinpath(pwd(), "outputs/fault")`).

## Running a simulation

```julia
run!(problem, time_stepper; outputs=outputs)
```

[`run!`](@ref) prints information about the simulation, the nonlinear residuals
at each time step, and a timing summary at the end (disable it with `log=false`).
Its keyword arguments control the nonlinear and linear solvers, the hierarchical
matrix compression and the preconditioner.

### Performance tips

- Start Julia with several threads (`julia -t auto`): the assembly of the
  hierarchical matrix, the matrix-vector products and the constraint updates are
  multithreaded.
- Since the matrix-vector products are already parallelized over the blocks of the
  hierarchical matrix, running BLAS on a single thread
  (`using LinearAlgebra; BLAS.set_num_threads(1)`) avoids oversubscribing the cores
  and is usually faster.
- The accuracy of the elastic matrix is controlled by `hmat_atol`; the solution is
  only as accurate as this compression tolerance.
- The preconditioner tolerance `pc_atol` and the refresh parameters `pc_refresh_it`
  and `pc_lin_rtol` only affect the speed of the solver, not the accuracy of the
  solution, which is controlled by the nonlinear tolerances `nl_abs_tol` and `nl_rel_tol`.
