using DDMFaultSlip
using StaticArrays
using SpecialFunctions
using LinearAlgebra
using FastGaussQuadrature
using DelimitedFiles
using Trapz

###### PHYSICAL PARAMETERS ######
const T = parse(Float64, ARGS[1])::Float64
const μ = 1.0::Float64 # Shear modulus
const h = 1.0e-05::Float64 # Fault thickness
const f = 1.0::Float64 # Friction coefficient

###### INITIAL VALUES ######
const σ₀ = 0.0::Float64 # Initial normal stress
const p₀ = 0.0::Float64 # Initial pressure

###### MESH PARAMETERS ######
const n = 10000::Int64 # Number of cells

###### TIME STEP PARAMETERS ######
const t₀ = 0.0::Float64 # Start time
const tₑ = 1.0::Float64 # End injection time
const nₜ = 500::Int64 # Number of injection time steps
const dt₀ = 1.0e-03::Float64 # Initial time step size

##### FUNCTIONS TO ESTIMATE λ ######
function residual_analytical(T::Float64, s::Vector{Float64}, w::Vector{Float64}, λ::Float64)
    f(s) = erfc(λ * abs(s))
    return dot(w, f.(s)) / pi - T
end

function jacobian_analytical(T::Float64, s::Vector{Float64}, w::Vector{Float64}, λ::Float64)
    f(s) = -2 * abs(s) / sqrt(pi) * exp(-λ^2 * abs(s)^2)
    return dot(w, f.(s)) / pi
end

function amplification_factor_2D(
    T::Float64;
    N::Int64=n,
    max_iters::Int64=200,
    abs_tol::Float64=1.0e-10,
    debug::Bool=false,
)
    # Initialization
    λ = 1.0
    k = 0
    # Create Gauss-Chebyshev quadrature points
    s, w = gausschebyshevt(N)

    while (k < max_iters)
        # Update residuals and jacobian
        Res = residual_analytical(T, s, w, λ)
        Jac = jacobian_analytical(T, s, w, λ)

        # Check convergence
        if (abs(Res) <= abs_tol)
            if (debug)
                println("Solve converged! λ = ", λ, " after ", k, " iterations.")
            end
            return λ
        end

        # Update λ
        dλ = -Res / Jac
        λ += dλ
        k += 1
    end

    throw(ErrorException("Maximm number of iterations reached in the Newton solve!!!\n"))
end

###### T-DEPENDENT PARAMETERS ######
# Length of the domain
function computeTDependentLength(T::Float64)::Float64
    # Analytical length of domain due to injection at constant rate
    λ = amplification_factor_2D(T)
    println("\tFault stress parameter : T = ", T)
    println("\tAmplification factor : λ = ", λ)
    println()

    return 1.1 * λ
end

###### FLUID PRESSURE ######
function fluid_pressure(X, time)
    return erfc.([norm(X[idx]) for idx in eachindex(X)] / sqrt(time))
end

function post_process_injection()
    # Read maximum file
    file = string(@__DIR__, "/outputs/constant_friction_2D_T_$(T).csv")
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
        file_t = string(@__DIR__, "/outputs/constant_friction_2D_T_$(T)_values/injection_$(i-1).csv")
        # Read data
        data_t = readdlm(file_t, ',', skipstart=1)
        # Unpack data
        r = data_t[:,1]
        σ = data_t[:,3]
        τ = data_t[:,6]
        δ = data_t[:,4]
        # Leading and trailing edge
        # Index where τ = f * σ (plastic yielding)
        idx = findall(isapprox.(τ, f * σ; atol=1.0e-03))
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
    ###### INITIAL VALUES ######
    τ₀ = -T::Float64 # background shear stress

    ###### INITIAL CONDITIONS FUNCTIONS ######
    function sigma_ic(X::Vector{SVector{2,Float64}})
        return σ₀ * ones(length(X))
    end

    function tau_ic(X::Vector{SVector{2,Float64}})
        return τ₀ * ones(length(X))
    end

    ###### DOMAIN LENGTH ######
    L = computeTDependentLength(T)::Float64 # Length of the domain

    ###### MESH ######
    start_point = SVector(-L, 0.0)
    end_point = SVector(L, 0.0)
    mesh = DDMesh1D(start_point, end_point, n)

    ###### PROBLEM ######
    problem = ShearDDProblem(mesh; μ=μ)

    ###### ICs ######
    addNormalStressIC!(problem, sigma_ic)
    addShearStressIC!(problem, tau_ic)

    ###### FLUID COUPLING ######
    addFluidCoupling!(problem, FunctionPressure(mesh, fluid_pressure))

    ###### FRICTION ######
    addFrictionConstraint!(problem, ConstantFriction(f, μ / h))

    ###### OUTPUTS ######
    outputs = [CSVDomainOutput(mesh, "outputs/constant_friction_2D_T_$(T)_values/injection"), CSVMaximumOutput("outputs/constant_friction_2D_T_$(T)")]

    ###### TIME SEQUENCE ######
    time_seq = round.(collect(exp10.(range(log10(t₀ + dt₀), stop=log10(tₑ), length=nₜ))); digits=6)
    time_stepper = TimeSequence(time_seq; start_time=t₀, end_time=tₑ)

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