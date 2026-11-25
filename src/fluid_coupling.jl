abstract type AbstractFluidCoupling end

struct DefaultFluidCoupling <: AbstractFluidCoupling
end

"""
    FunctionPressure(mesh::DDMesh{T}, fct::Function; fct_ic::Function=default_fc_ic)

Fluid coupling with a pore pressure prescribed by the function `fct(X, time)`,
which returns the vector of pressures at the element centroids `X` at time `time`.
The change of pressure reduces the effective normal stress acting on the
fault (``σ' = σ - p``). `fct_ic(X)` gives the initial pressure (zero by default).

# Example
```julia
# Pressure diffusing from a line source at x = 0
pressure(X, t) = Δp * erfc.([abs(x[1]) for x in X] / sqrt(α * t))
addFluidCoupling!(problem, FunctionPressure(mesh, pressure))
```
"""
mutable struct FunctionPressure{T<:Real} <: AbstractFluidCoupling
    " Pressure value"
    p::Vector{T}

    " Pressure old value"
    p_old::Vector{T}

    " Function to udate pressure"
    fct::Function

    " Constructor"
    function FunctionPressure(mesh::DDMesh{T}, fct::Function; fct_ic::Function=default_fc_ic) where {T<:Real}
        # Initial pressure value
        p = fct_ic([mesh.elems[i].X for i in eachindex(mesh.elems)])
        # Old pressure value
        p_old = copy(p)

        return new{T}(p, p_old, fct)
    end
end

function updatePressure!(fc::FunctionPressure{T}, X, time::T) where {T<:Real}
    fc.p = fc.fct(X, time)
    return nothing
end

function default_fc_ic(X::Vector{Point2D{T}}) where {T<:Real}
    return zeros(length(X))
end

function default_fc_ic(X::Vector{Point3D{T}}) where {T<:Real}
    return zeros(length(X))
end