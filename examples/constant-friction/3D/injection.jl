using DDMFaultSlip
using StaticArrays
using SpecialFunctions
using LinearAlgebra
using HypergeometricFunctions
using Roots

###### PHYSICAL PARAMETERS ######
const T = parse(Float64, ARGS[1])::Float64
const μ = 1.0::Float64 # Shear modulus
const h = 1.0e-05::Float64 # Fault thickness
const f = 1.0::Float64 # Friction coefficient

###### INITIAL VALUES ######
const σ₀ = 0.0::Float64 # Initial normal stress
const p₀ = 0.0::Float64 # Initial pressure

###### TIME STEP PARAMETERS ######
const t₀ = 0.0::Float64 # Start time
const tₑ = 1.0::Float64 # End injection time
const nₜ = 500::Int64 # Number of injection time steps
const dt₀ = 1.0e-03::Float64 # Initial time step size

##### FUNCTIONS TO ESTIMATE λ ######
function stress_parameter_3D(λ::Float64)
    return 2.0 - Base.MathConstants.γ + 2.0 * λ^2 / 3.0 * pFq([1.0, 1.0], [2.0, 2.5], -λ^2) - log(4.0 * λ^2)
end
function amplification_factor_3D(T::Float64)
    return abs(fzero(λ->stress_parameter_3D(λ) - T, 0.01))
end

###### T-DEPENDENT PARAMETERS ######
# Length of the domain
function computeTDependentLength(T::Float64)::Float64
    # Analytical length of domain due to injection at constant rate
    λ = amplification_factor_3D(T)
    println("\tFault stress parameter : T = ", T)
    println("\tAmplification factor : λ = ", λ)
    println()

    return 1.1 * λ
end

###### FLUID PRESSURE ######
function fluid_pressure_injection(X, time::T)::Vector{T} where {T<:Real}
    return expint.([norm(X[idx])^2 for idx in eachindex(X)] / time)
end

function fluid_pressure(X, time::T)::Vector{T} where {T<:Real}
    if (time > 0.0)
        return fluid_pressure_injection(X, time)
    else
        return 0.0
    end
end

function main()
    ###### INITIAL VALUES ######
    τ₀ = -T::Float64 # Initial shear stress

    ###### INITIAL CONDITIONS FUNCTIONS ######
    function sigma_ic(X)
        return σ₀ * ones(length(X))
    end

    function tau_ic(X)
        return τ₀ * ones(length(X))
    end

    ###### DOMAIN LENGTH ######
    R = computeTDependentLength(T)::Float64 # Length of the domain

    ###### MESH ######
    # Create mesh
    gmsh_command = `gmsh -2 $(@__DIR__)/mesh.geo -setnumber R $(R) -o $(@__DIR__)/mesh-$(T).msh`
    run(gmsh_command)
    mesh = DDMesh2D(Float64, "mesh-$(T).msh")

    ###### PROBLEM ######
    problem = ShearDDProblem(mesh; axisymmetric=true)

    ###### ICs ######
    addNormalStressIC!(problem, sigma_ic)
    addShearStressIC!(problem, tau_ic)

    ###### FLUID COUPLING ######
    addFluidCoupling!(problem, FunctionPressure(mesh, fluid_pressure))

    ###### FRICTION ######
    addFrictionConstraint!(problem, ConstantFriction(f, μ / h))

    ###### OUTPUTS ######
    outputs = [VTKDomainOutput(mesh, "outputs/injection_values_T_$(T)/injection_T_$(T)"), CSVMaximumOutput("outputs/injection_T_$(T)")]

    ###### TIME SEQUENCE ######
    time_seq = round.(collect(exp10.(range(log10(t₀ + dt₀), stop=log10(tₑ), length=nₜ))); digits=6)
    time_stepper = TimeSequence(time_seq; start_time=t₀, end_time=tₑ)

    ###### RUN PROBLEM ######
    run!(problem, time_stepper; outputs = outputs, output_initial = true)
end

main()