abstract type AbstractConstraint end

function computeConstraints(cst::AbstractConstraint, Δu::T, X::SVector{2,T})::Tuple{T,T} where {T<:Real}
    throw(MethodError(computeConstraints, cst))
end

function computeConstraints(cst::AbstractConstraint, Δu::T, X::SVector{3,T})::Tuple{T,T} where {T<:Real}
    throw(MethodError(computeConstraints, cst))
end

struct DefaultConstraint <: AbstractConstraint
end

function computeConstraints(cst::DefaultConstraint, Δu::T, X::SVector{2,T})::Tuple{T,T} where {T<:Real}
    return (0.0, 0.0)
end

function computeConstraints(cst::DefaultConstraint, Δu::T, X::SVector{3,T})::Tuple{T,T} where {T<:Real}
    return (0.0, 0.0)
end

struct FunctionConstraint <: AbstractConstraint
    " Residual function"
    func_res::Function

    " Jacobian function"
    func_jac::Function

    " Constructors"
    function FunctionConstraint(func_res::Function)
        return new(func_res, (Δu, X) -> 0.0)
    end

    function FunctionConstraint(func_res::Function, func_jac::Function)
        return new(func_res, func_jac)
    end
end

function computeConstraints(cst::FunctionConstraint, Δu::T, X::SVector{2,T})::Tuple{T,T} where {T<:Real}
    return (cst.func_res(X, Δu), cst.func_jac(X, Δu))
end

function computeConstraints(cst::FunctionConstraint, Δu::T, X::SVector{3,T})::Tuple{T,T} where {T<:Real}
    return (cst.func_res(X, Δu), cst.func_jac(X, Δu))
end
