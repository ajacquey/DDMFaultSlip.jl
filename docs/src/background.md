# [Background](@id background-section)

This page summarizes the formulation implemented in `DDMFaultSlip.jl`.

## The Displacement Discontinuity Method

A fault or fracture is a surface ``Γ`` across which the displacement field of the
surrounding elastic medium is discontinuous. The displacement discontinuity (DD)
has a normal component, the opening ``w``, and a tangential component, the slip
``δ``. In an infinite, homogeneous, linear elastic medium, the tractions acting on
``Γ`` depend linearly on the DD through boundary integrals.

The Displacement Discontinuity Method discretizes ``Γ`` into ``N`` elements, each
carrying a uniform (piecewise constant) DD. The tractions are evaluated at the
element centroids (collocation points), which gives

```math
σ_i = σ_i^0 + \sum_{j=1}^{N} E_{ij}\, w_j,
\qquad
τ_i = τ_i^0 + \sum_{j=1}^{N} E_{ij}\, δ_j,
```

where ``σ^0`` and ``τ^0`` are the initial normal and shear stresses, and the
influence coefficients ``E_{ij}`` are the stress at the centroid of element ``i``
due to a unit DD on element ``j``. Normal stresses are positive in compression.
`DDMFaultSlip.jl` uses closed-form influence coefficients for:

| Mesh | Geometry | Elements |
|:-----|:---------|:---------|
| [`DDMesh1D`](@ref) | 2D plane strain | straight segments |
| [`DDMesh1D`](@ref) with `axisymmetric=true` | 3D axisymmetric | rings (radial profile) |
| [`DDMesh2D`](@ref) | 3D planar fault | triangles |

For 3D shear problems with a non-zero Poisson's ratio ``ν``, the two slip
components are coupled and both are solved for. With ``ν = 0``, they decouple and a
single component is solved for.

## Hierarchical matrices

The elastic matrix ``E`` is dense: storing it and multiplying it by a vector costs
``O(N^2)``. `DDMFaultSlip.jl` stores it as a hierarchical matrix using
[HMatrices.jl](https://github.com/IntegralEquations/HMatrices.jl). The element
centroids are clustered, and blocks of ``E`` coupling well-separated clusters
(controlled by the admissibility parameter `hmat_eta`) are compressed to low-rank
approximations with adaptive cross approximation (tolerance `hmat_atol`). Storage
and matrix-vector products then scale roughly as ``O(N \log N)``, which makes large
3D meshes tractable.

## Interface laws

The behaviour of the fault or fracture is described by a local relation between
the traction and the DD of each element. Frictional faults are modelled as a thin
elasto-plastic interface: the slip increment ``Δδ`` is split into an elastic and a
plastic (frictional) part, ``Δδ = Δδ^e + Δδ^p``, and

```math
τ = τ_{old} + k \left(Δδ - Δδ^p\right),
\qquad
F(τ, σ', Δδ^p) = τ - f\, σ' - η \frac{Δδ^p}{Δt} \le 0,
```

where ``k`` is the shear stiffness of the interface, ``f`` the friction coefficient,
``σ' = σ - p`` the effective normal stress, and ``η`` an optional viscosity. The
stiffness ``k`` is typically taken as ``μ / h`` with ``h`` a small fault thickness, so
that the elastic slip is negligible: it acts as a penalty enforcing stick where
the fault does not slide.

For each element, the plastic slip is computed with a return-mapping algorithm:
a trial traction ``τ^{tr} = τ_{old} + k Δδ`` is computed, and if ``F(τ^{tr}) > 0``, the
plastic slip increment ``Δδ^p`` is found such that ``F = 0``. The available laws are:

- [`ConstantFriction`](@ref): constant friction coefficient ``f``.
- [`SlipWeakeningFriction`](@ref): friction decreasing linearly with slip from
  ``f_p`` to ``f_r`` over the slip ``δ_r``.
- [`ConstantYield`](@ref): constant shear strength, independent of ``σ'``.
- [`DugdaleCohesiveZone`](@ref): cohesive zone for opening problems.
- [`FunctionConstraint`](@ref): a user-defined stress change as a function of the
  position and the DD increment.

## Fluid coupling

A pore pressure ``p(\mathbf{x}, t)`` acting on the fault reduces the effective normal
stress, ``σ' = σ - p``, and therefore the frictional strength ``f σ'``. With
[`FunctionPressure`](@ref), the pressure is prescribed as a function of space and
time, for instance an analytical solution of pore pressure diffusion from an
injection point. Pressure increases can then trigger frictional slip, which
redistributes the shear stress elastically along the fault.

## Nonlinear solver

At each time step, the DD increments ``Δu`` (opening or slip) of all elements must
satisfy simultaneously the elastic relation and the interface laws:

```math
R_i(Δu) = \sum_{j=1}^{N} E_{ij}\, Δu_j - Δt_i(Δu_i) = 0,
```

where ``Δt_i`` is the traction increment given by the interface law of element
``i``. This nonlinear system is solved with Newton's method. Its Jacobian is the
elastic matrix plus a (block) diagonal matrix of local tangent stiffnesses,
``J = E + D``, so only ``D`` changes between iterations. Each linear system is
solved with a Krylov method (IDR(s) by default, GMRES or BiCGStab(l)) from
[IterativeSolvers.jl](https://github.com/JuliaLinearAlgebra/IterativeSolvers.jl),
preconditioned with a hierarchical LU factorization of ``J``. The factorization is
expensive, so it is reused across nonlinear iterations and time steps and only
recomputed when the linear solver needs too many iterations or does not reach
an accurate solution.

The problems are quasi-static: time enters through the prescribed pore pressure,
the viscosity ``η`` and the evolution of the interface laws. The time steps are
controlled by a time stepper ([`ConstantDT`](@ref) or [`TimeSequence`](@ref)), and a
time step for which the Newton solver fails is retried with half its size.
