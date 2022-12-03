abstract type AbstractVariable{T<:Real} end

mutable struct Variable{T<:Real} <: AbstractVariable{T}
    " The value of the variable"
    value::Vector{T}

    " The old value of the variable"
    value_old::Vector{T}

    " The symbol for the variable"
    sym::Symbol

    " The function describing the initial conditions"
    func_ic::Function

    " Constructor"
    function Variable(T::Type, sym::Symbol, n::Int)
        return new{T}(zeros(T, n), zeros(T, n), sym, default_ic)
    end
end

mutable struct AuxVariable{T<:Real} <: AbstractVariable{T}
    " The value of the auxiliary variable"
    value::Vector{T}

    " The old value of the auxiliary variable"
    value_old::Vector{T}

    " The symbol for the variable"
    sym::Symbol

    " The function describing the initial conditions"
    func_ic::Function

    " Constructor"
    function AuxVariable(T::Type, sym::Symbol, n::Int)
        return new{T}(zeros(T, n), zeros(T, n), sym, default_ic)
    end
end

function default_ic(X::Vector{Point2D{T}}) where {T<:Real}
    return zeros(length(X))
end

function default_ic(X::Vector{Point3D{T}}) where {T<:Real}
    return zeros(length(X))
end