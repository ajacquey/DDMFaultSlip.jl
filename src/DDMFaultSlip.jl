module DDMFaultSlip

using StaticArrays
using LinearAlgebra
using Statistics: mean
using InteractiveUtils
using Printf
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
export FunctionConstraint

include("constraints/friction.jl")
export ConstantYield
export ConstantFriction

include("fluid_coupling.jl")
export FunctionPressure

include("problem.jl")
export NormalDDProblem, ShearDDProblem2D, ShearDDProblem3D, CoupledDDProblem2D, CoupledDDProblem3D
export addNormalStressIC!, addShearStressIC!
export addConstraint!
export addFrictionConstraint!
export addFluidCoupling!

include("jacobian.jl")

include("solver.jl")

include("assembly.jl")

include("executioner.jl")
export run!
end
