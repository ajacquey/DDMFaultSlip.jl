module DDMFaultSlip

using StaticArrays
using LinearAlgebra
using Statistics: mean
using HMatrices
using TimerOutputs

include("mesh.jl")
export Point2D, Point3D, Mesh1D, Mesh2D

include("collocation.jl")

include("jacobian.jl")

include("solver.jl")
export DirectDDMSolver, IterativeDDMSolver
end
