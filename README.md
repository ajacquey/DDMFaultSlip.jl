# DDMFaultSlip

*A package for modelling fault and fracture mechanics using the Displacement Discontinuity Method (DDM)*

[![License](https://img.shields.io/github/license/mashape/apistatus.svg?maxAge=2592000)](https://github.com/ajacquey/DDMFaultSlip.jl/blob/master/LICENSE)
[![Build Status](https://github.com/ajacquey/DDMFaultSlip.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/ajacquey/DDMFaultSlip.jl/actions/workflows/CI.yml?query=branch%3Amain)
[![Coverage](https://codecov.io/gh/ajacquey/DDMFaultSlip.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/ajacquey/DDMFaultSlip.jl)

## Documentation

[![](https://img.shields.io/badge/docs-latest%20release-blue)](https://ajacquey.github.io/DDMFaultSlip.jl/stable/)

## Installation

You can install `DDMFaultSlip` directly from the `Pkg` REPL:

```
pkg> add DDMFaultSlip
```

## Related package
`DDMFaultSlip` relies on the following essential packages:
* [HMatrices.jl](https://github.com/IntegralEquations/HMatrices.jl): used for hierarchical matrices handling.
* [IterativeSolvers.jl](https://github.com/JuliaLinearAlgebra/IterativeSolvers.jl): used for solving linear systems.