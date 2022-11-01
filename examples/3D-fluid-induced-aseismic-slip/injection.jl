using DDMFaultSlip
using StaticArrays
using SpecialFunctions
using LinearAlgebra

###### PHYSICAL PARAMETERS ######
const T = parse(Float64, ARGS[1])::Float64
const λ = 6.6667e+02::Float64
const μ = 6.6667e+02::Float64
const ν = 0.0::Float64
const h = 1.0e-03::Float64
const f = 0.5::Float64
const σ₀ = 1.0::Float64
const Δp = 0.4::Float64
const α = 0.04::Float64
const t₀ = 0.0::Float64 # start time
const tₑ = 600.0::Float64 # end time
const dt₀ = 1.0::Float64 # initial time step size
###### NUMERICAL PARAMETERS ######
const nₜ = 500::Int64

###### T-DEPENDENT PARAMETERS ######
# Background shear stress
function computeTDependentShearStress(T::Float64)::Float64
    return f * σ₀ * (1.0 - T * Δp / σ₀)
end

const τ₀ = computeTDependentShearStress(T)::Float64 # background shear stress

###### INITIAL CONDITIONS ######
function sigma_ic(X)
    return σ₀
end

function tau_x_ic(X)
    return τ₀
end

function tau_y_ic(X)
    return 0.0
end

###### FLUID PRESSURE ######
function fluid_pressure(X, time)
    return Δp * expint(1, norm(X)^2 / (4 * α * time))
end

function main()
    ###### MESH ######
    mesh = DDMesh2D(Float64, "mesh.msh")

    ###### PROBLEM ######
    problem = CoupledDDProblem3D(mesh; μ=μ, ν=ν)

    ###### ICs ######
    addNormalStressIC!(problem, sigma_ic)
    addShearStressIC!(problem, SVector(tau_x_ic, tau_y_ic))

    ###### FLUID COUPLING ######
    addFluidCoupling!(problem, FunctionPressure(mesh, fluid_pressure))

    ###### FRICTION ######
    addFrictionConstraint!(problem, ConstantFriction(f, μ / h, μ / h))

    ###### OUTPUTS ######
    outputs = [VTKDomainOutput(mesh, "outputs/injection_values_T_$(T)/injection_T_$(T)")]

    ###### TIME SEQUENCE ######
    # time_seq = collect(exp10.(range(log10(t₀+dt₀), stop=log10(tₑ), length=nₜ)))
    time_seq = collect(range(0.0, stop=tₑ, length=nₜ))
    # Quick fix
    time_seq[end] = tₑ
    time_stepper = TimeSequence(time_seq; start_time=0.0, end_time=tₑ)

    ###### RUN PROBLEM ######
    run!(problem, time_stepper; outputs=outputs, output_initial=true)
end

main()