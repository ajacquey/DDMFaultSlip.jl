abstract type AbstractDDProblem{T<:Real} end

mutable struct NormalDDProblem{T<:Real} <: AbstractDDProblem{T}
    " The mesh for the problem"
    mesh::DDMesh{T}

    " The elastic shear modulus"
    μ::Float64

    " The Poisson's ratio"
    ν::Float64

    " A boolean to check if system is initialized"
    initialized::Bool

    " A boolean to specify if the problem is transient"
    transient::Bool

    " The normal DD variable"
    ϵ::Variable{T}

    " The normal DD stress"
    σ::AuxVariable{T}

    " A vector of Constraints"
    constraints::Vector{AbstractConstraint}

    " Constructor"
    function NormalDDProblem(mesh::DDMesh{T}; transient::Bool=false, μ::T=1.0, ν::T=0.0) where {T<:Real}
        return new{T}(mesh, μ, ν, false, transient, Variable(T, :ϵ, length(mesh.elems)), AuxVariable(T, :σ, length(mesh.elems)), Vector{AbstractConstraint}(undef, 0))
    end
end

mutable struct ShearDDProblem2D{T<:Real} <: AbstractDDProblem{T}
    " The mesh for the problem"
    mesh::DDMesh1D{T}

    " The elastic shear modulus"
    μ::Float64

    " A boolean to check if system is initialized"
    initialized::Bool

    " A boolean to specify if the problem is transient"
    transient::Bool

    " The shear DD variables"
    δ::Variable{T}

    " The shear DD stress"
    τ::AuxVariable{T}

    " A vector of Constraints"
    constraints::Vector{AbstractConstraint}

    " Incomplete constructor"
    function ShearDDProblem2D(mesh::DDMesh1D{T}; transient::Bool=false, μ::T=1.0) where {T<:Real}
        return new{T}(mesh, μ, false, transient, Variable(T, :δ, length(mesh.elems)), AuxVariable(T, :τ, length(mesh.elems)), Vector{AbstractConstraint}(undef, 0))
    end
end

mutable struct ShearDDProblem3D{T<:Real} <: AbstractDDProblem{T}
    " The mesh for the problem"
    mesh::DDMesh2D{T}

    " The elastic shear modulus"
    μ::Float64

    " The Poisson's ratio"
    ν::Float64

    " A boolean to check if system is initialized"
    initialized::Bool

    " A boolean to specify if the problem is transient"
    transient::Bool

    " The shear DD variables"
    δ_x::Variable{T}
    δ_y::Variable{T}

    " The shear DD stress variables"
    τ_x::AuxVariable{T}
    τ_y::AuxVariable{T}

    " A vector of vector of Constraints"
    constraints_x::Vector{AbstractConstraint}
    constraints_y::Vector{AbstractConstraint}

    " Incomplete constructor"
    function ShearDDProblem3D(mesh::DDMesh2D{T}; transient::Bool=false, μ::T=1.0, ν::T=0.0) where {T<:Real}
        return new{T}(mesh, μ, ν, false, transient, Variable(T, :δ_x, length(mesh.elems)), Variable(T, :δ_y, length(mesh.elems)), AuxVariable(T, :τ_x, length(mesh.elems)), AuxVariable(T, :τ_y, length(mesh.elems)), Vector{AbstractConstraint}(undef, 0), Vector{AbstractConstraint}(undef, 0))
    end
end

mutable struct CoupledDDProblem2D{T<:Real} <: AbstractDDProblem{T}
    " The mesh for the problem"
    mesh::DDMesh1D{T}

    " The elastic shear modulus"
    μ::Float64

    " A boolean to check if system is initialized"
    initialized::Bool

    " A boolean to specify if the problem is transient"
    transient::Bool

    " The normal DD variable"
    ϵ::Variable{T}

    " The shear DD variables"
    δ::Variable{T}

    " The normal DD stress"
    σ::AuxVariable{T}

    " The shear DD stress"
    τ::AuxVariable{T}

    " Incomplete constructor"
    function CoupledDDProblem2D(mesh::DDMesh{T}; transient::Bool=false, μ::T=1.0, ν::T=0.0) where {T<:Real}
        return new{T}(mesh, μ, false, transient, Variable(T, length(mesh.elems)), Variable(T, length(mesh.elems)), AuxVariable(T, length(mesh.elems)), AuxVariable(T, length(mesh.elems)))
    end
end

mutable struct CoupledDDProblem3D{T<:Real} <: AbstractDDProblem{T}
    " The mesh for the problem"
    mesh::DDMesh2D{T}

    " The elastic shear modulus"
    μ::Float64

    " The Poisson's ratio"
    ν::Float64

    " A boolean to check if system is initialized"
    initialized::Bool

    " A boolean to specify if the problem is transient"
    transient::Bool

    " The normal DD variable"
    ϵ::Variable{T}

    " The shear DD variables"
    δ::SVector{2, Variable{T}}

    " The normal DD stress"
    σ::AuxVariable{T}

    " The shear DD stress"
    τ::SVector{2,AuxVariable{T}}

    " Incomplete constructor"
    function CoupledDDProblem(mesh::DDMesh{T}; transient::Bool=false, μ::T=1.0, ν::T=0.0) where {T<:Real}
        return new{T}(mesh, μ, ν, false, transient, Variable(T, length(mesh.elems)), SVector(Variable(T, length(mesh.elems)), Variable(T, length(mesh.elems))), AuxVariable(T, length(mesh.elems)), SVector(AuxVariable(T, length(mesh.elems)), AuxVariable(T, length(mesh.elems))))
    end
end

function addNormalDDIC!(problem::NormalDDProblem{T}, func_ic::Function) where {T<:Real}
    problem.ϵ.func_ic = func_ic
    return nothing
end

function addShearDDIC!(problem::ShearDDProblem2D{T}, func_ic::Function) where {T<:Real}
    problem.δ.func_ic = func_ic
    return nothing
end

function addShearDDIC!(problem::ShearDDProblem3D{T}, func_ic::SVector{2,Function}) where {T<:Real}
    problem.δ_x.func_ic = func_ic[1]
    problem.δ_y.func_ic = func_ic[2]
    return nothing
end

function applyNormalDDIC!(problem::AbstractDDProblem{T}) where {T<:Real}
    problem.ϵ.value = problem.ϵ.func_ic.([problem.mesh.elems[i].X for i in 1:length(problem.mesh.elems)])
    problem.ϵ.value_old = copy(problem.ϵ.value)
    problem.σ.value = problem.σ.func_ic.([problem.mesh.elems[i].X for i in 1:length(problem.mesh.elems)])
    problem.σ.value_old = copy(problem.σ.value)
    return nothing
end

function applyShearDDIC!(problem::AbstractDDProblem{T}) where {T<:Real}
    if isa(problem, ShearDDProblem2D)
        problem.δ.value = problem.δ.func_ic.([problem.mesh.elems[i].X for i in 1:length(problem.mesh.elems)])
        problem.δ.value_old = copy(problem.δ.value)
        problem.τ.value = problem.τ.func_ic.([problem.mesh.elems[i].X for i in 1:length(problem.mesh.elems)])
        problem.τ.value_old = copy(problem.τ.value)
    elseif isa(problem, ShearDDProblem3D)
        problem.δ_x.value = problem.δ_x.func_ic.([problem.mesh.elems[i].X for i in 1:length(problem.mesh.elems)])
        problem.δ_x.value_old = copy(problem.δ_x.value)
        problem.τ_x.value = problem.τ_x.func_ic.([problem.mesh.elems[i].X for i in 1:length(problem.mesh.elems)])
        problem.τ_x.value_old = copy(problem.τ_x.value)
        problem.δ_y.value = problem.δ_y.func_ic.([problem.mesh.elems[i].X for i in 1:length(problem.mesh.elems)])
        problem.δ_y.value_old = copy(problem.δ_y.value)
        problem.τ_y.value = problem.τ_y.func_ic.([problem.mesh.elems[i].X for i in 1:length(problem.mesh.elems)])
        problem.τ_y.value_old = copy(problem.τ_y.value)
    else
        throw(ErrorException("No shear IC in this problem!"))
    end
    return nothing
end

function applyIC!(problem::AbstractDDProblem{T}) where {T<:Real}
    if isa(problem, NormalDDProblem)
        applyNormalDDIC!(problem)
    elseif isa(problem, ShearDDProblem2D) || isa(problem, ShearDDProblem3D)
        applyShearDDIC!(problem)
    end
    # Need to look at CoupledProblems
    return nothing
end

function addConstraint!(problem::NormalDDProblem{T}, cst::AbstractConstraint) where {T<:Real}
    push!(problem.constraints, cst)
    return nothing
end

function addConstraint!(problem::ShearDDProblem2D{T}, cst::AbstractConstraint) where {T<:Real}
    push!(problem.constraints, cst)
    return nothing
end


function addConstraint!(problem::ShearDDProblem3D{T}, sym::Symbol, cst::AbstractConstraint) where {T<:Real}
    if (sym == :x)
        push!(problem.constraints_x, cst)
    elseif (sym == :y)
        push!(problem.constraints_y, cst)
    else
        throw(ErrorException("No dimension noted $(sym)!"))
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
