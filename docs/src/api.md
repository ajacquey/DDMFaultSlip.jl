```@meta
CurrentModule = DDMFaultSlip
```

# [API reference](@id api-section)

## Meshes

```@docs
Point2D
Point3D
DDMesh1D
DDMesh2D
```

## Problems

```@docs
NormalDDProblem
ShearDDProblem
addNormalDDIC!
addShearDDIC!
addNormalStressIC!
addShearStressIC!
```

## Constraints

```@docs
addConstraint!
FunctionConstraint
addFrictionConstraint!
ConstantFriction
SlipWeakeningFriction
ConstantYield
addCohesiveZoneConstraint!
DugdaleCohesiveZone
```

## Fluid coupling

```@docs
addFluidCoupling!
FunctionPressure
```

## Time stepping

```@docs
ConstantDT
TimeSequence
```

## Outputs

```@docs
VTKDomainOutput
CSVDomainOutput
CSVMaximumOutput
```

## Elastic matrices

```@docs
DD3DShearElasticMatrix
```

## Running simulations

```@docs
run!
DDSolver
```

## Index

```@index
Pages = ["api.md"]
```
