function F(e::Point3D{T}, r::Point3D{T}) where {T<:Real}
    return (r[1] * e[1] + r[2] * e[2]) / norm(r)
end

function integralI003(ei::DDTriangleElem{T}, ej::DDTriangleElem{T}) where {T<:Real}
    I = 0.0
    for p in 1:3
        ap = ej.nodes[p]
        if p < 3
            bp = ej.nodes[p+1]
        else
            bp = ej.nodes[1]
        end
        ra = ei.X - ap
        rb = ei.X - bp
        ep = bp - ap
        I += 1.0 / (ra[2] * ep[1] - ra[1] * ep[2]) * (F(ep, rb) - F(ep, ra))
    end
    return I
end

function integralI205(ei::DDTriangleElem{T}, ej::DDTriangleElem{T}) where {T<:Real}
    I = 0.0
    for p in 1:3
        ap = ej.nodes[p]
        if p < 3
            bp = ej.nodes[p+1]
        else
            bp = ej.nodes[1]
        end
        ra = ei.X - ap
        rb = ei.X - bp
        ep = bp - ap
        I += (ra[2] * ep[1] - 2.0 * ra[1] * ep[2]) / (3 * (ra[2] * ep[1] - ra[1] * ep[2])^2) * (F(ep, rb) - F(ep, ra)) + ep[1] * ep[2] / (3 * (ra[2] * ep[1] - ra[1] * ep[2])^2) * (F(ra, rb) - F(ra, ra))
    end
    return I
end

function integralI025(ei::DDTriangleElem{T}, ej::DDTriangleElem{T}) where {T<:Real}
    I = 0.0
    for p in 1:3
        ap = ej.nodes[p]
        if p < 3
            bp = ej.nodes[p+1]
        else
            bp = ej.nodes[1]
        end
        ra = ei.X - ap
        rb = ei.X - bp
        ep = bp - ap
        I += (2.0 * ra[2] * ep[1] - ra[1] * ep[2]) / (3 * (ra[2] * ep[1] - ra[1] * ep[2])^2) * (F(ep, rb) - F(ep, ra)) - ep[1] * ep[2] / (3 * (ra[2] * ep[1] - ra[1] * ep[2])^2) * (F(ra, rb) - F(ra, ra))
    end
    return I
end

function integralI115(ei::DDTriangleElem{T}, ej::DDTriangleElem{T}) where {T<:Real}
    I = 0.0
    for p in 1:3
        ap = ej.nodes[p]
        if p < 3
            bp = ej.nodes[p+1]
        else
            bp = ej.nodes[1]
        end
        ra = ei.X - ap
        rb = ei.X - bp
        ep = bp - ap
        I += (ra[1] * ep[1] - ra[2] * ep[2]) / (6 * (ra[2] * ep[1] - ra[1] * ep[2])^2) * (F(ep, rb) - F(ep, ra)) + (ep[2]^2 - ep[1]^2) / (6 * (ra[2] * ep[1] - ra[1] * ep[2])^2) * (F(ra, rb) - F(ra, ra))
    end
    return I
end

abstract type ElasticKernelMatrix{T<:Real} <: AbstractMatrix{T} end

"""
Piecewise constant (PWC) two-dimensional elastic kernel matrix
"""
struct DD2DElasticMatrix{T<:Real} <: ElasticKernelMatrix{T}
    e::Vector{DDEdgeElem{T}}
    μ::T
end

function Base.getindex(K::DD2DElasticMatrix, i::Int, j::Int)
    return 2.0 * K.μ * norm((K.e[j].nodes[2] - K.e[j].nodes[1]) / 2.0) / (π * (norm(K.e[i].X - K.e[j].X)^2 - norm((K.e[j].nodes[2] - K.e[j].nodes[1]) / 2.0)^2))
end

function Base.size(K::DD2DElasticMatrix)
    return length(K.e), length(K.e)
end

"""
Piecewise constant (PWC) three-dimensional normal elastic kernel matrix
"""
struct DD3DNormalElasticMatrix{T<:Real} <: ElasticKernelMatrix{T}
    e::Vector{DDTriangleElem{T}}
    μ::T
    ν::T
end

function Base.getindex(K::DD3DNormalElasticMatrix, i::Int, j::Int)
    return K.μ / (4 * π * (1.0 - K.ν)) * integralI003(K.e[i], K.e[j])
end

function Base.size(K::DD3DNormalElasticMatrix)
    return length(K.e), length(K.e)
end

"""
Piecewise constant (PWC) three-dimensional shear elastic kernel matrix XX
"""
struct DD3DShearElasticMatrixXX{T<:Real} <: ElasticKernelMatrix{T}
    e::Vector{DDTriangleElem{T}}
    μ::T
    ν::T
end

function Base.getindex(K::DD3DShearElasticMatrixXX, i::Int, j::Int)
    return K.μ / (4 * π * (1.0 - K.ν)) * ((1.0 - 2 * K.ν) * integralI003(K.e[i], K.e[j]) + 3.0 * K.ν * integralI205(K.e[i], K.e[j]))
end

function Base.size(K::DD3DShearElasticMatrixXX)
    return length(K.e), length(K.e)
end

"""
Piecewise constant (PWC) three-dimensional shear elastic kernel matrix YY
"""
struct DD3DShearElasticMatrixYY{T<:Real} <: ElasticKernelMatrix{T}
    e::Vector{DDTriangleElem{T}}
    μ::T
    ν::T
end

function Base.getindex(K::DD3DShearElasticMatrixYY, i::Int, j::Int)
    return K.μ / (4 * π * (1.0 - K.ν)) * ((1.0 - 2 * K.ν) * integralI003(K.e[i], K.e[j]) + 3.0 * K.ν * integralI025(K.e[i], K.e[j]))
end

function Base.size(K::DD3DShearElasticMatrixYY)
    return length(K.e), length(K.e)
end

"""
Piecewise constant (PWC) three-dimensional shear elastic kernel matrix XY
"""
struct DD3DShearElasticMatrixXY{T<:Real} <: ElasticKernelMatrix{T}
    e::Vector{DDTriangleElem{T}}
    μ::T
    ν::T
end

function Base.getindex(K::DD3DShearElasticMatrixXY, i::Int, j::Int)
    return K.μ / (4 * π * (1.0 - K.ν)) * 3.0 * K.ν * integralI115(K.e[i], K.e[j])
end

function Base.size(K::DD3DShearElasticMatrixXY)
    return length(K.e), length(K.e)
end

"""
Piecewise constant (PWC) three-dimensional elastic kernel matrix for ν = 0
"""
struct DD3DElasticMatrix{T<:Real} <: ElasticKernelMatrix{T}
    e::Vector{DDTriangleElem{T}}
    μ::T
end

function Base.getindex(K::DD3DElasticMatrix, i::Int, j::Int)
    return K.μ / (4 * π) * integralI003(K.e[i], K.e[j])
end

function Base.size(K::DD3DElasticMatrix)
    return length(K.e), length(K.e)
end

"""
Piecewise constant (PWC) three-dimensional shear elastic kernel matrix
"""
struct DD3DShearElasticMatrix{T<:Real} <: ElasticKernelMatrix{T}
    e::Vector{DDTriangleElem{T}}
    μ::T
    ν::T
end

function Base.getindex(K::DD3DShearElasticMatrix, i::Int, j::Int)
    n = div(length(K.e), 2)
    if i <= n
        if j <= n
            return K.μ / (4 * π * (1.0 - K.ν)) * ((1.0 - 2 * K.ν) * integralI003(K.e[i], K.e[j]) + 3.0 * K.ν * integralI205(K.e[i], K.e[j]))
        else
            if K.ν == 0.0
                return 0.0
            else
                return 3.0 * K.ν * K.μ / (4 * π * (1.0 - K.ν)) * integralI115(K.e[i], K.e[j])
            end
        end
    else
        if j <= n
            if K.ν == 0.0
                return 0.0
            else
                return 3.0 * K.ν * K.μ / (4 * π * (1.0 - K.ν)) * integralI115(K.e[i], K.e[j])
            end
        else
            return K.μ / (4 * π * (1.0 - K.ν)) * ((1.0 - 2 * K.ν) * integralI003(K.e[i], K.e[j]) + 3.0 * K.ν * integralI025(K.e[i], K.e[j]))
        end
    end
end

function Base.size(K::DD3DShearElasticMatrix)
    return length(K.e), length(K.e)
end

"""
Piecewise constant (PWC) three-dimensional shear elastic kernel matrix with ν = 0
"""
struct DD3DShearNoNuElasticMatrix{T<:Real} <: ElasticKernelMatrix{T}
    e::Vector{DDTriangleElem{T}}
    μ::T
end

function Base.getindex(K::DD3DShearNoNuElasticMatrix, i::Int, j::Int)
    return K.μ / (4 * π) * integralI003(K.e[i], K.e[j])
end

function Base.size(K::DD3DShearNoNuElasticMatrix)
    return length(K.e), length(K.e)
end