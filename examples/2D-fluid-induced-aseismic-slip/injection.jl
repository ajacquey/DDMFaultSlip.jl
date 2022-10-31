using DDMFaultSlip
using StaticArrays
using SpecialFunctions
using LinearAlgebra

###### PHYSICAL PARAMETERS ######
const T = parse(Float64, ARGS[1])::Float64
const λ = 6.6667e+02::Float64
const μ = 6.6667e+02::Float64
const h = 1.0e-03::Float64
const f = 0.5::Float64
const σ₀ = 1.0::Float64
const Δp = 0.4::Float64
const α = 0.04::Float64
const t₀ = 0.0::Float64 # start time
const tₑ = 600.0::Float64 # end time
const dt₀ = 0.01::Float64 # initial time step size
###### NUMERICAL PARAMETERS ######
const n = 10000::Int64
const nₜ = 500::Int64
const order = 0::Int64

###### T-DEPENDENT PARAMETERS ######
# Length of the domain
function computeTDependentLength(T::Float64)::Float64
    # # Read table in file
    # data = readdlm(string(@__DIR__, "/data_T_dependent.csv"), ',', skipstart=1)
    # idx = findall(x -> x == T, data[:,1])
    # return data[idx[1], 3]
    return 10.0
end
# Background shear stress
function computeTDependentShearStress(T::Float64)::Float64
    return f * σ₀ * (1.0 - T * Δp / σ₀)
end

const L = computeTDependentLength(T)::Float64 # length of the domain
const τ₀ = computeTDependentShearStress(T)::Float64 # background shear stress

###### INITIAL CONDITIONS ######
function sigma_ic(X)
    return σ₀
end

function tau_ic(X)
    return τ₀
end

###### FLUID PRESSURE ######
function fluid_pressure(X, time)
    return Δp * erfc(norm(X) / sqrt(α * time))
end

function main()
    ###### MESH ######
    start_point = SVector(-L / 2.0, 0.0)
    end_point = SVector(L / 2.0, 0.0)
    mesh = DDMesh1D(start_point, end_point, n)

    ###### PROBLEM ######
    problem = CoupledDDProblem2D(mesh; μ=μ)

    ###### ICs ######
    addNormalStressIC!(problem, sigma_ic)
    addShearStressIC!(problem, tau_ic)

    ###### FLUID COUPLING ######
    addFluidCoupling!(problem, FunctionPressure(mesh, fluid_pressure))

    ###### FRICTION ######
    addFrictionConstraint!(problem, ConstantFriction(f, μ / h, μ / h))

    ###### OUTPUTS ######
    outputs = [CSVDomainOutput(mesh, "outputs/injection_values_T_$(T)/injection_T_$(T)")]

    ###### TIME SEQUENCE ######
    time_seq = collect(exp10.(range(log10(t₀+dt₀), stop=log10(tₑ), length=nₜ)))
    # Quick fix
    time_seq[end] = tₑ
    time_stepper = TimeSequence(time_seq; start_time=0.0, end_time=tₑ)

    ###### RUN PROBLEM ######
    run!(problem, time_stepper; outputs=outputs)
end

main()