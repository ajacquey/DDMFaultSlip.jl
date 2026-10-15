using DDMFaultSlip
using StaticArrays
using SpecialFunctions
using LinearAlgebra
using DelimitedFiles
using Trapz

# Parameters
const fᵣ_fₚ = parse(Float64, ARGS[1])::Float64
const Δp_σ₀ = parse(Float64, ARGS[2])::Float64
const τ₀_τₚ = parse(Float64, ARGS[3])::Float64
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
const tₑ = 250.0::Float64 # end time
const nₜ = 1000::Int # number of time steps
const n = 1000::Int # number of elements

# Initial stress conditions
function σ_ic(X)
    return σ₀ * ones(length(X))
end

function τ_ic(X)
    return τ₀ * ones(length(X))
end

# Fluid pressure
function fluid_pressure(X, time)
    return erfc.([abs(X[idx][1]) for idx in eachindex(X)] / sqrt(time))
end

function friction(δ)
    δᵣ = 1.0 - fᵣ_fₚ
    if δ < δᵣ
        return 1.0 - δ
    else
        return fᵣ_fₚ
    end
end

function post_process_injection()
    # Read maximum file
    file = string(@__DIR__, "/outputs/slip_weakening_friction_2D_f_$(fᵣ_fₚ)_p_$(Δp_σ₀)_tau_$(τ₀_τₚ).csv")
    # Read data
    data = readdlm(file, ',')
    num_columns = size(data, 2)
    num_lines = size(data, 1) - 1
    # Declare rupture front
    Rᵣ = zeros(num_lines)
    # Declare scalar moment release
    M₀ = zeros(num_lines)
    # Loop over time steps
    Threads.@threads for i in eachindex(Rᵣ)
        # CSV file with domain output
        file_t = string(@__DIR__, "/outputs/slip_weakening_friction_2D_f_$(fᵣ_fₚ)_p_$(Δp_σ₀)_tau_$(τ₀_τₚ)_values/injection_$(i-1).csv")
        # Read data
        data_t = readdlm(file_t, ',', skipstart=1)
        # Unpack data
        r = data_t[:,1]
        σ = data_t[:,3]
        τ = data_t[:,6]
        δ = data_t[:,4]
        # Leading and trailing edge
        # Index where τ = f * σ (plastic yielding)
        idx = findall(isapprox.(τ, friction.(δ) .* σ; atol=1.0e-03))
        if length(idx) > 0
            Rᵣ[i] = r[idx[end]]
        else
            Rᵣ[i] = 0.0
        end
        M₀[i] = 2 * π * μ * trapz(r, r .* δ)
    end
    println("Done computing rupture front!")
    println()
    
    # Create new vectors
    new_column_1 = Vector{Any}()
    new_column_2 = Vector{Any}()
    # Add columns header
    push!(new_column_1, "R_r")
    push!(new_column_2, "M_0")
    # Add leading and trailing edge and moment release data
    append!(new_column_1, Rᵣ)
    append!(new_column_2, M₀)
    if (num_columns == 7) # first time running this script
        println("Adding new column to file: ", file)
        # Merge new column with existing data
        data = hcat(data, new_column_1, new_column_2)
    else # Rerunning script
        println("Updating existing column in file: ", file)
        data[:, end-1] .= new_column_1
        data[:,end] .= new_column_2
    end
    # Write new data to file
    open(file; write=true) do f
        writedlm(f, data, ',')
    end
    return
end

function run_injection()
    # Create mesh
    R = 20.0
    ###### MESH ######
    start_point = SVector(-R, 0.0)
    end_point = SVector(R, 0.0)
    mesh = DDMesh1D(start_point, end_point, n)

    # Create problem
    problem = ShearDDProblem(mesh; μ=μ, ν=ν)

    # Initial stress conditions
    addNormalStressIC!(problem, σ_ic)
    addShearStressIC!(problem, τ_ic)

    # Fluid coupling
    addFluidCoupling!(problem, FunctionPressure(mesh, fluid_pressure))

    # Friction
    addFrictionConstraint!(problem, SlipWeakeningFriction(1.0, fᵣ_fₚ, 1.0 - fᵣ_fₚ, k; η=0.01*μ))

    ###### OUTPUTS ######
    outputs = [CSVDomainOutput(mesh, "outputs/slip_weakening_friction_2D_f_$(fᵣ_fₚ)_p_$(Δp_σ₀)_tau_$(τ₀_τₚ)_values/injection"), CSVMaximumOutput("outputs/slip_weakening_friction_2D_f_$(fᵣ_fₚ)_p_$(Δp_σ₀)_tau_$(τ₀_τₚ)")]
    
    ###### TIME SEQUENCE ######
    time_stepper = ConstantDT(t₀, tₑ, nₜ)

    ###### RUN PROBLEM ######
    run!(problem, time_stepper; output_initial = true, outputs = outputs)
end

function main()
    # Run simulation
    run_injection()

    # Post-process results
    post_process_injection()
end

main()