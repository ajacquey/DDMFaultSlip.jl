abstract type AbstractConstraint end

function computeConstraints(cst::AbstractConstraint, time::T, X::SVector{2, T})::Tuple{T, T} where {T<:Real}
    throw(MethodError(computeConstraints, cst))
end

function computeConstraints(cst::AbstractConstraint, time::T, X::SVector{3, T})::Tuple{T, T} where {T<:Real}
    throw(MethodError(computeConstraints, cst))
end