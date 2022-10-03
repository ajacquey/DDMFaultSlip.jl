module DDMFaultSlip

using StaticArrays
using LinearAlgebra
using Statistics: mean
using HMatrices
using IterativeSolvers
using TimerOutputs
using WriteVTK

include("mesh.jl")
export Point2D, Point3D, DDMesh1D, DDMesh2D

include("collocation.jl")
export DD3DShearElasticMatrix

include("time_stepper.jl")
export TimeSequence

include("variable.jl")

include("constraints/constraint.jl")
include("constraints/function_constraint.jl")
export FunctionConstraint

include("problem.jl")
export NormalDDProblem, ShearDDProblem2D, ShearDDProblem3D, CoupledDDProblem
export addNormalDDVariable!, addShearDDVariable!
export addConstraint!

include("jacobian.jl")

include("solver.jl")

include("assembly.jl")

include("executioner.jl")
export run!
end
