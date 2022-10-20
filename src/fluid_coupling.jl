abstract type AbstractFluidCoupling{T<:Real} end

mutable struct FunctionPressure{T<:Real} <: AbstractFluidCoupling{T}
    " Pressure value"
    p::Vector{T}

    " Pressure old value"
    p_old::Vector{T}

    " Function to udate pressure"
    fct::Function

    " Constructor"
    function FunctionPressure(mesh::DDMesh{T}, fct::Function; fct_ic::Function = default_fc_ic) where {T<:Real}
        # Initial pressure value
        p = fct_ic.([mesh.elems[i].X for i in 1:length(mesh.elems)])
        # Old pressure value
        p_old = copy(p)

        return new{T}(p, p_old, fct)
    end
end

function updatePressure!(fc::FunctionPressure{T}, X, time::T, idx::Int) where {T<:Real}
    fc.p[idx] = fc.fct(X, time)
    return nothing
end

function default_fc_ic(X::Point2D{T}) where {T<:Real}
    return 0.0
end

function default_fc_ic(X::Point3D{T}) where {T<:Real}
    return 0.0
end