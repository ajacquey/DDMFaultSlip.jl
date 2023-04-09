abstract type AbstractDDProblem{T<:Real} end

mutable struct NormalDDProblem{T<:Real} <: AbstractDDProblem{T}
    " The mesh for the problem"
    mesh::DDMesh{T}

    " The elastic shear modulus"
    μ::Float64

    " The Poisson's ratio"
    ν::Float64

    " The number of elements"
    n::Int

    " The number of degrees of freedom"
    n_dof::Int

    " A boolean to specify if the problem is transient"
    transient::Bool

    " The normal DD variable"
    w::Variable{T}

    " The normal stress"
    σ::AuxVariable{T}

    " Constraints"
    constraints::AbstractConstraint

    " Frictional constraint"
    friction::AbstractFriction

    " Cohesive zone"
    cohesive::AbstractCohesiveZone

    " Pressure coupling"
    fluid_coupling::AbstractFluidCoupling

    " Constructor"
    function NormalDDProblem(mesh::DDMesh{T}; transient::Bool=false, μ::T=1.0, ν::T=0.0) where {T<:Real}
        return new{T}(mesh, μ, ν, length(mesh.elems), length(mesh.elems), transient,
            Variable(T, :w, length(mesh.elems)),
            AuxVariable(T, :σ, length(mesh.elems)),
            DefaultConstraint(),
            DefaultFriction(),
            DefaultCohesiveZone(),
            DefaultFluidCoupling(),
        )
    end
end

mutable struct ShearDDProblem{T<:Real} <: AbstractDDProblem{T}
    " The mesh for the problem"
    mesh::DDMesh{T}

    " The elastic shear modulus"
    μ::Float64

    " The Poisson's ratio"
    ν::Float64

    " The number of elements"
    n::Int

    " The number of degrees of freedom"
    n_dof::Int

    " A boolean to specify if the problem is transient"
    transient::Bool

    " The shear DD variables"
    δ::Variable{T}

    " The normal stress"
    σ::AuxVariable{T}

    " The shear stress"
    τ::AuxVariable{T}

    " A vector of Constraints"
    constraints::Vector{AbstractConstraint}

    " Frictional constraint"
    friction::AbstractFriction

    " Cohesive zone"
    cohesive::AbstractCohesiveZone

    " Pressure coupling"
    fluid_coupling::AbstractFluidCoupling

    " Constructor"
    function ShearDDProblem(mesh::DDMesh{T}; transient::Bool=false, μ::T=1.0, ν::T=0.0) where {T<:Real}
        if isa(mesh, DDMesh1D)
            return new{T}(mesh, μ, ν,  length(mesh.elems), length(mesh.elems), transient,
                Variable(T, :δ, length(mesh.elems)),
                AuxVariable(T, :σ, length(mesh.elems)),
                AuxVariable(T, :τ, length(mesh.elems)),
                [DefaultConstraint()],
                DefaultFriction(),
                DefaultCohesiveZone(),
                DefaultFluidCoupling(),
            )
        else
            if (ν == 0.0)
                return new{T}(mesh, μ, ν,  length(mesh.elems), length(mesh.elems), transient,
                    Variable(T, :δ, length(mesh.elems)),
                    AuxVariable(T, :σ, length(mesh.elems)),
                    AuxVariable(T, :τ, length(mesh.elems)),
                    [DefaultConstraint()],
                    DefaultFriction(),
                    DefaultCohesiveZone(),
                    DefaultFluidCoupling(),
                )
            else
                return new{T}(mesh, μ, ν,  length(mesh.elems), 2*length(mesh.elems), transient,
                    Variable(T, :δ, 2*length(mesh.elems)),
                    AuxVariable(T, :σ, length(mesh.elems)),
                    AuxVariable(T, :τ, 2*length(mesh.elems)),
                    [DefaultConstraint(), DefaultConstraint()],
                    DefaultFriction(),
                    DefaultCohesiveZone(),
                    DefaultFluidCoupling(),
                )
            end
        end
    end
end

function addNormalDDIC!(problem::AbstractDDProblem{T}, func_ic::Function) where {T<:Real}
    if hasproperty(problem, :w)
        problem.w.func_ic = func_ic
    else
        throw(ErrorException("This problem doesn't have a normal DD!"))
    end
    return nothing
end

function addShearDDIC!(problem::AbstractDDProblem{T}, func_ic::Function) where {T<:Real}
    if hasproperty(problem, :δ)
        problem.δ.func_ic = func_ic
    else
        throw(ErrorException("This problem doesn't have a shear DD!"))
    end
    return nothing
end

function addNormalStressIC!(problem::AbstractDDProblem{T}, func_ic::Function) where {T<:Real}
    if hasproperty(problem, :σ)
        problem.σ.func_ic = func_ic
    else
        throw(ErrorException("This problem doesn't have a normal stress!"))
    end
    return nothing
end

function addShearStressIC!(problem::AbstractDDProblem{T}, func_ic::Function) where {T<:Real}
    if hasproperty(problem, :τ)
        problem.τ.func_ic = func_ic
    else
        throw(ErrorException("This problem doesn't have a shear shear!"))
    end
    return nothing
end

function applyNormalDDIC!(problem::AbstractDDProblem{T}) where {T<:Real}
    problem.w.value = problem.w.func_ic([problem.mesh.elems[i].X for i in eachindex(problem.mesh.elems)])
    problem.w.value_old = copy(problem.w.value)
    problem.σ.value = problem.σ.func_ic([problem.mesh.elems[i].X for i in eachindex(problem.mesh.elems)])
    problem.σ.value_old = copy(problem.σ.value)
    return nothing
end

function applyShearDDIC!(problem::AbstractDDProblem{T}) where {T<:Real}
    if (problem.n == problem.n_dof) # 1D mesh or 2D axis symmetric 
        problem.δ.value = problem.δ.func_ic([problem.mesh.elems[i].X for i in eachindex(problem.mesh.elems)])
        problem.δ.value_old = copy(problem.δ.value)
        problem.τ.value = problem.τ.func_ic([problem.mesh.elems[i].X for i in eachindex(problem.mesh.elems)])
        problem.τ.value_old = copy(problem.τ.value)
    else
        problem.δ.value[1:problem.n] = problem.δ.func_ic([problem.mesh.elems[i].X for i in eachindex(problem.mesh.elems)], :x)
        problem.δ.value_old[1:problem.n] = copy(problem.δ.value[1:problem.n])
        problem.δ.value[problem.n+1:2*problem.n] = problem.δ.func_ic([problem.mesh.elems[i].X for i in eachindex(problem.mesh.elems)], :y)
        problem.δ.value_old[problem.n+1:2*problem.n] = copy(problem.δ.value[problem.n+1:2*problem.n])
        problem.τ.value[1:problem.n] = problem.τ.func_ic([problem.mesh.elems[i].X for i in eachindex(problem.mesh.elems)], :x)
        problem.τ.value_old[1:problem.n] = copy(problem.τ.value[1:problem.n])
        problem.τ.value[problem.n+1:2*problem.n] = problem.τ.func_ic([problem.mesh.elems[i].X for i in eachindex(problem.mesh.elems)], :y)
        problem.τ.value_old[problem.n+1:2*problem.n] = copy(problem.τ.value[problem.n+1:2*problem.n])
    end
    problem.σ.value = problem.σ.func_ic([problem.mesh.elems[i].X for i in eachindex(problem.mesh.elems)])
    problem.σ.value_old = copy(problem.σ.value)
    return nothing
end

function applyIC!(problem::AbstractDDProblem{T}) where {T<:Real}
    if isa(problem, NormalDDProblem)
        applyNormalDDIC!(problem)
    elseif isa(problem, ShearDDProblem)
        applyShearDDIC!(problem)
    end
    return nothing
end

function hasConstraint(problem::AbstractDDProblem{T})::Bool where {T<:Real}
    if isa(problem.constraints, Vector)
        return ~all(x->isa(x, DefaultConstraint), problem.constraints)
    else
        return ~isa(problem.constraints, DefaultConstraint)
    end
end

function addConstraint!(problem::NormalDDProblem{T}, cst::AbstractConstraint) where {T<:Real}
    problem.constraints = cst
    return nothing
end

function addConstraint!(problem::ShearDDProblem{T}, cst::AbstractConstraint) where {T<:Real}
    if (problem.n_dof > problem.n) # 2D shear
        throw(ErrorException("For 3D problem, please specify direction as a symbol: :x or :y."))
    end
    problem.constraints[1] = cst
    return nothing
end

function addConstraint!(problem::ShearDDProblem{T}, cst::AbstractConstraint, sym::Symbol) where {T<:Real}
    if isa(problem.mesh, DDMesh1D)
        throw(ErrorException("For 2D problem, you don't need to specify the direction!"))
    end
    if (sym == :x)
        problem.constraints[1] =  cst
    elseif (sym == :y)
        problem.constraints[2] =  cst
    else
        throw(ErrorException("No dimension noted $(sym)! Possible dimensions are ':x', or ':y'"))
    end
    return nothing
end

function hasFrictionConstraint(problem::AbstractDDProblem{T})::Bool where {T<:Real}
    return ~isa(problem.friction, DefaultFriction)
end

function addFrictionConstraint!(problem::AbstractDDProblem{T}, friction::AbstractFriction) where {T<:Real}
    # Check if problem has friction
    if (hasFrictionConstraint(problem))
        throw(ErrorException("The problem already has a FrictionConstraint!"))
    end

    # Add FrictionConstraint
    problem.friction = friction

    return nothing
end

function hasCohesiveZoneConstraint(problem::AbstractDDProblem{T})::Bool where {T<:Real}
    return ~isa(problem.cohesive, DefaultCohesiveZone)
end

function addCohesiveZoneConstraint!(problem::AbstractDDProblem{T}, cohesive::AbstractCohesiveZone) where {T<:Real}
    # Check if problem has cohesive
    if (hasCohesiveZoneConstraint(problem))
        throw(ErrorException("The problem already has a CohesiveZoneConstraint!"))
    end

    # Add CohesiveZoneConstraint
    problem.cohesive = cohesive

    return nothing
end

function hasFluidCoupling(problem::AbstractDDProblem{T})::Bool where {T<:Real}
    return ~isa(problem.fluid_coupling, DefaultFluidCoupling)
end

function addFluidCoupling!(problem::AbstractDDProblem{T}, pp::AbstractFluidCoupling) where {T<:Real}
    # Check if FluidCoupling is not empty
    if (hasFluidCoupling(problem))
        throw(ErrorException("The problem already has a FluidCoupling!"))
    end

    # Add FluidCoupling
    problem.fluid_coupling = pp

    return nothing
end

function reinit!(problem::AbstractDDProblem{T}) where {T<:Real}
    # Normal DD and stress
    if hasproperty(problem, :w)
        problem.w.value_old = copy(problem.w.value)
    end
    if hasproperty(problem, :σ)
        problem.σ.value_old = copy(problem.σ.value)
    end
    # Shear DD and stress
    if hasproperty(problem, :δ)
        problem.δ.value_old = copy(problem.δ.value)
    end
    if hasproperty(problem, :τ)
        problem.τ.value_old = copy(problem.τ.value)
    end
    # Fluid coupling
    if hasFluidCoupling(problem)
        problem.fluid_coupling.p_old = copy(problem.fluid_coupling.p)
    end
    return nothing
end

function Base.show(io::IO, problem::AbstractDDProblem{T}) where {T<:Real}
    # Julia information
    versioninfo()
    @printf("\n")

    # Paralelism
    @printf("Paralelism:\n")
    @printf("  Num threads: %i\n\n", Threads.nthreads())

    # Mesh
    show(problem.mesh)
end
