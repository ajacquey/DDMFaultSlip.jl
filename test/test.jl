using DDMFaultSlip
using StaticArrays
using SpecialFunctions
using LinearAlgebra

# Fault stress parameter
const T = parse(Float64, ARGS[1])::Float64
# Elastic properties
const μ = 1.0::Float64
const ν = 0.0::Float64
const h = 1.0e-05::Float64
const k = (μ / h)::Float64
# Friction
const f = 0.5::Float64
const σ₀ = 1.0::Float64
# Fluid properties
const α = 0.04::Float64
const Δp = 0.15
# Time 
const t₀ = 0.0::Float64 # start time
const tₑ = 600.0::Float64 # end time
const dt₀ = 1.0::Float64 # initial time step size
const nₜ = 50::Int # number of time steps
# Initial shear stress
const τ₀ = (f * σ₀ * (1 - Δp * T))::Float64

# Initial stress conditions
function σ_ic(X)
    return σ₀ * ones(length(X))
end

function τ_ic(X)
    return τ₀ * ones(length(X))
end

# Fluid pressure
function fluid_pressure(X, time)
    return Δp * expint.(1, [norm(X[idx])^2 for idx in eachindex(X)] / (α * time))
end

function main()
    # Create mesh
    R = 3 * sqrt(α * tₑ)
    gmsh_command = `gmsh -2 meshes/mesh-inj.geo -setnumber R $(R) -o meshes/mesh-constant-friction-$(T).msh`
    run(gmsh_command)
    mesh = DDMesh2D(Float64, "meshes/mesh-constant-friction-$(T).msh")

    # Create problem
    problem = ShearDDProblem(mesh; μ=μ, ν=ν)

    # Initial stress conditions
    addNormalStressIC!(problem, σ_ic)
    addShearStressIC!(problem, τ_ic)

    # Fluid coupling
    addFluidCoupling!(problem, FunctionPressure(mesh, fluid_pressure))

    # Friction
    addFrictionConstraint!(problem, ConstantFriction(f, k))

    ###### OUTPUTS ######
    outputs = [VTKDomainOutput(mesh, "outputs/constant_friction_T_$(T)_values/injection_T_$(T)"), CSVMaximumOutput("outputs/constant_friction_T_$(T)")]

    ###### TIME SEQUENCE ######
    time_seq = collect(exp10.(range(log10(t₀ + dt₀), stop=log10(tₑ), length=nₜ)))
    # Quick fix
    time_seq[end] = tₑ
    time_stepper = TimeSequence(time_seq; start_time=0.0, end_time=tₑ)
    # time_seq = collect(range(0.0, stop=2.0, length=3))
    # time_stepper = TimeSequence(time_seq; start_time=0.0, end_time=2.0)

    ###### RUN PROBLEM ######
    # run!(problem, time_stepper; log=true, linear_log=true, outputs=outputs, output_initial=true, nl_abs_tol=1.0e-08, pc=true, pc_atol=1.0e-01)
    run!(problem, time_stepper; log=true, linear_log=true, outputs=outputs, output_initial=true, nl_abs_tol=1.0e-08, pc=false, l_solver="gmres")
end

main()