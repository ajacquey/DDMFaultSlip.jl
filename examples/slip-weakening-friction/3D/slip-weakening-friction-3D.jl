using DDMFaultSlip
using StaticArrays
using SpecialFunctions
using LinearAlgebra

# Parameters
# const fᵣ_fₚ = parse(Float64, ARGS[1])::Float64
# const Δp_σ₀ = parse(Float64, ARGS[2])::Float64
# const τ₀_τₚ = parse(Float64, ARGS[3])::Float64
const fᵣ_fₚ = 0.7
const Δp_σ₀ = 0.0355
const τ₀_τₚ = 0.6
# Elastic properties
const μ = (1 / Δp_σ₀)::Float64
const ν = 0.0::Float64
const h = 1.0e-05::Float64
const k = μ / h::Float64
# Initial normal stress
const σ₀ = (1 / Δp_σ₀)::Float64
# Initial shear stress
const τ₀ = (τ₀_τₚ / Δp_σ₀)::Float64
# Time 
const t₀ = 0.0::Float64 # start time
const tₑ = 160000.0::Float64 # end time
const nₜ = 2000::Int # number of time steps
const R = 25.0::Float64

# Initial stress conditions
function σ_ic(X)
    return σ₀ * ones(length(X))
end

function τ_ic(X)
    return τ₀ * ones(length(X))
end

# Fluid pressure
function fluid_pressure(X, time)
    return expint.(1, [norm(X[idx])^2 for idx in eachindex(X)] / time)
end

function main()
    # Create mesh
    gmsh_command = `gmsh -2 $(@__DIR__)/../../meshes/mesh.geo -setnumber R $(R) -o $(@__DIR__)/mesh-P$(Δp_σ₀).msh`
    run(gmsh_command)
    mesh = DDMesh2D(Float64, "mesh-P$(Δp_σ₀).msh")
    

    # Create problem
    problem = ShearDDProblem(mesh; μ=μ, ν=ν, axisymmetric=true)
   
    # Initial stress conditions
    addNormalStressIC!(problem, σ_ic)
    addShearStressIC!(problem, τ_ic)

    # Fluid coupling
    addFluidCoupling!(problem, FunctionPressure(mesh, fluid_pressure))

    # Friction
    addFrictionConstraint!(problem, SlipWeakeningFriction(1.0, fᵣ_fₚ, 1.0 - fᵣ_fₚ, k; η=0.01*μ))

    ###### OUTPUTS ######
     outputs = [
        VTKDomainOutput(mesh, "outputs/sw3D_F$(fᵣ_fₚ)_P$(Δp_σ₀)_S$(τ₀_τₚ)/injection"),
        CSVMaximumOutput("outputs/sw3D_F$(fᵣ_fₚ)_P$(Δp_σ₀)_S$(τ₀_τₚ)"),
    ]

     ###### TIME SEQUENCE ######
    time_stepper = ConstantDT(t₀, tₑ, nₜ)
    
    ###### RUN PROBLEM ######
    # run!(problem, time_stepper; outputs=outputs,
    #      output_initial=true,
    #      nl_abs_tol=1.0e-06, nl_max_it=20,
    #      pc=true, pc_atol=1.0e-02,
    #      l_solver="idrs", linear_log=true,
    #      l_rel_tol=1.0e-04, l_abs_tol=1.0e-12, l_max_it=1000,
    #      dt_min=1.0e-03)
    run!(problem, time_stepper; outputs=outputs,
         output_initial=true,
         nl_abs_tol=1.0e-06, nl_max_it=20)
end

main()