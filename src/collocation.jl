function F(e::Point3D{T}, r::Point3D{T})::T where {T<:Real}
    return (r[1] * e[1] + r[2] * e[2]) / norm(r)
end

function integralI003(ei::DDTriangleElem{T}, ej::DDTriangleElem{T})::T where {T<:Real}
    I = 0.0
    for p in 1:3
        ap = ej.nodes[p]
        if p < 3
            bp = ej.nodes[p+1]
        else
            bp = ej.nodes[1]
        end
        I += 1.0 / ((ei.X - ap)[2] * (bp - ap)[1] - (ei.X - ap)[1] * (bp - ap)[2]) * (F(bp - ap, ei.X - bp) - F(bp - ap, ei.X - ap))
    end
    return I
end

function integralI205(ei::DDTriangleElem{T}, ej::DDTriangleElem{T})::T where {T<:Real}
    I = 0.0
    for p in 1:3
        ap = ej.nodes[p]
        if p < 3
            bp = ej.nodes[p+1]
        else
            bp = ej.nodes[1]
        end
        I += ((ei.X - ap)[2] * (bp - ap)[1] - 2.0 * (ei.X - ap)[1] * (bp - ap)[2]) / (3 * ((ei.X - ap)[2] * (bp - ap)[1] - (ei.X - ap)[1] * (bp - ap)[2])^2) * (F(bp - ap, ei.X - bp) - F(bp - ap, ei.X - ap)) + (bp - ap)[1] * (bp - ap)[2] / (3 * ((ei.X - ap)[2] * (bp - ap)[1] - (ei.X - ap)[1] * (bp - ap)[2])^2) * (F(ei.X - ap, ei.X - bp) - F(ei.X - ap, ei.X - ap))
    end
    return I
end

function integralI025(ei::DDTriangleElem{T}, ej::DDTriangleElem{T})::T where {T<:Real}
    I = 0.0
    for p in 1:3
        ap = ej.nodes[p]
        if p < 3
            bp = ej.nodes[p+1]
        else
            bp = ej.nodes[1]
        end
        I += (2.0 * (ei.X - ap)[2] * (bp - ap)[1] - (ei.X - ap)[1] * (bp - ap)[2]) / (3 * ((ei.X - ap)[2] * (bp - ap)[1] - (ei.X - ap)[1] * (bp - ap)[2])^2) * (F(bp - ap, ei.X - bp) - F(bp - ap, ei.X - ap)) - (bp - ap)[1] * (bp - ap)[2] / (3 * ((ei.X - ap)[2] * (bp - ap)[1] - (ei.X - ap)[1] * (bp - ap)[2])^2) * (F(ei.X - ap, ei.X - bp) - F(ei.X - ap, ei.X - ap))
    end
    return I
end

function integralI115(ei::DDTriangleElem{T}, ej::DDTriangleElem{T})::T where {T<:Real}
    I = 0.0
    for p in 1:3
        ap = ej.nodes[p]
        if p < 3
            bp = ej.nodes[p+1]
        else
            bp = ej.nodes[1]
        end
        I += ((ei.X - ap)[1] * (bp - ap)[1] - (ei.X - ap)[2] * (bp - ap)[2]) / (6 * ((ei.X - ap)[2] * (bp - ap)[1] - (ei.X - ap)[1] * (bp - ap)[2])^2) * (F(bp - ap, ei.X - bp) - F(bp - ap, ei.X - ap)) + ((bp - ap)[2]^2 - (bp - ap)[1]^2) / (6 * ((ei.X - ap)[2] * (bp - ap)[1] - (ei.X - ap)[1] * (bp - ap)[2])^2) * (F(ei.X - ap, ei.X - bp) - F(ei.X - ap, ei.X - ap))
    end
    return I
end

function m(ri::Point2D{T}, rj::Point2D{T}) where {T<:Real}
    return 4.0 * norm(ri) * norm(rj) / (norm(ri) + norm(rj))^2
end

abstract type ElasticKernelMatrix{T<:Real} <: AbstractMatrix{T} end

"""
Piecewise constant (PWC) two-dimensional elastic kernel matrix
"""
struct DD2DElasticMatrix{T<:Real} <: ElasticKernelMatrix{T}
    e::Vector{DDEdgeElem{T}}
    μ::T
    n::Int

    # Constructor
    function DD2DElasticMatrix(mesh::DDMesh1D{T}, μ::T) where {T<:Real}
        return new{T}(mesh.elems, μ, length(mesh.elems))
    end
end

function Base.getindex(K::DD2DElasticMatrix, i::Int, j::Int)
    return 2.0 * K.μ * norm((K.e[j].nodes[2] - K.e[j].nodes[1]) / 2.0) / (π * (norm(K.e[i].X - K.e[j].X)^2 - norm((K.e[j].nodes[2] - K.e[j].nodes[1]) / 2.0)^2))
end

function Base.size(K::DD2DElasticMatrix)
    return K.n, K.n
end

"""
Piecewise constant (PWC) three-dimensional axisymmetric elastic kernel matrix
"""
struct DD3DAxisymmetricElasticMatrix{T<:Real} <: ElasticKernelMatrix{T}
    e::Vector{DDEdgeElem{T}}
    μ::T
    n::Int

    # Constructor
    function DD3DAxisymmetricElasticMatrix(mesh::DDMesh1D{T}, μ::T) where {T<:Real}
        return new{T}(mesh.elems, μ, length(mesh.elems))
    end
end

function Base.getindex(K::DD3DAxisymmetricElasticMatrix, i::Int, j::Int)
    return K.μ / π * (((norm(K.e[j].nodes[2]) - norm(K.e[i].X)) * ellipe(m(K.e[i].X, K.e[j].nodes[1])) - (norm(K.e[j].nodes[1]) - norm(K.e[i].X)) * ellipe(m(K.e[i].X, K.e[j].nodes[2]))) / (norm(K.e[i].X - K.e[j].X)^2 - norm((K.e[j].nodes[2] - K.e[j].nodes[1]) / 2.0)^2) + ((norm(K.e[j].nodes[2]) + norm(K.e[i].X)) * ellipk(m(K.e[i].X, K.e[j].nodes[1])) - (norm(K.e[j].nodes[1]) + norm(K.e[i].X)) * ellipk(m(K.e[i].X, K.e[j].nodes[2]))) / (norm(K.e[i].X + K.e[j].X)^2 - norm((K.e[j].nodes[2] - K.e[j].nodes[1]) / 2.0)^2))
end

function Base.size(K::DD3DAxisymmetricElasticMatrix)
    return K.n, K.n
end

"""
Piecewise constant (PWC) three-dimensional normal elastic kernel matrix
"""
struct DD3DNormalElasticMatrix{T<:Real} <: ElasticKernelMatrix{T}
    e::Vector{DDTriangleElem{T}}
    μ::T
    ν::T
    n::Int

    # Constructor
    function DD3DNormalElasticMatrix(mesh::DDMesh2D{T}, μ::T, ν::T) where {T<:Real}
        return new{T}(mesh.elems, μ, ν, length(mesh.elems))
    end
end

function Base.getindex(K::DD3DNormalElasticMatrix, i::Int, j::Int)
    return K.μ / (4 * π * (1.0 - K.ν)) * integralI003(K.e[i], K.e[j])
end

function Base.size(K::DD3DNormalElasticMatrix)
    return K.n, K.n
end

"""
Piecewise constant (PWC) three-dimensional shear elastic kernel matrix
"""
struct DD3DShearElasticMatrix{T<:Real} <: ElasticKernelMatrix{T}
    e::Vector{DDTriangleElem{T}}
    μ::T
    ν::T
    n::Int

    # Constructor
    function DD3DShearElasticMatrix(mesh::DDMesh2D{T}, μ::T, ν::T) where {T<:Real}
        return new{T}(mesh.elems, μ, ν, length(mesh.elems))
    end
end

function Base.getindex(K::DD3DShearElasticMatrix, i::Int, j::Int)
    if i <= K.n
        if j <= K.n
            return K.μ / (4 * π * (1.0 - K.ν)) * ((1.0 - 2 * K.ν) * integralI003(K.e[i], K.e[j]) + 3.0 * K.ν * integralI205(K.e[i], K.e[j]))
        else
            return 3.0 * K.ν * K.μ / (4 * π * (1.0 - K.ν)) * integralI115(K.e[i], K.e[j-K.n])
        end
    else
        if j <= K.n
            return 3.0 * K.ν * K.μ / (4 * π * (1.0 - K.ν)) * integralI115(K.e[i-K.n], K.e[j])
        else
            return K.μ / (4 * π * (1.0 - K.ν)) * ((1.0 - 2 * K.ν) * integralI003(K.e[i-K.n], K.e[j-K.n]) + 3.0 * K.ν * integralI025(K.e[i-K.n], K.e[j-K.n]))
        end
    end
end

function Base.size(K::DD3DShearElasticMatrix)
    return 2 * K.n, 2 * K.n
end

"""
Piecewise constant (PWC) three-dimensional shear axis symmetric elastic kernel matrix
"""
struct DD3DShearAxisymmetricElasticMatrix{T<:Real} <: ElasticKernelMatrix{T}
    e::Vector{DDTriangleElem{T}}
    μ::T
    n::Int

    # Constructor
    function DD3DShearAxisymmetricElasticMatrix(mesh::DDMesh2D{T}, μ::T) where {T<:Real}
        return new{T}(mesh.elems, μ, length(mesh.elems))
    end
end

function Base.getindex(K::DD3DShearAxisymmetricElasticMatrix, i::Int, j::Int)
    return K.μ / (4 * π) * integralI003(K.e[i], K.e[j])
end

function Base.size(K::DD3DShearAxisymmetricElasticMatrix)
    return K.n, K.n
end

"""
Piecewise constant (PWC) two-dimensional jacobian matrix
"""
struct DD2DJacobianMatrix{T<:Real} <: ElasticKernelMatrix{T}
    e::Vector{DDEdgeElem{T}}
    mat_loc::Matrix{Vector{T}}
    μ::T
    n::Int

    # Constructor
    function DD2DJacobianMatrix(mesh::DDMesh1D{T}, mat_loc::Matrix{Vector{T}}, μ::T) where {T<:Real}
        return new{T}(mesh.elems, mat_loc, μ, length(mesh.elems))
    end
end

function Base.getindex(K::DD2DJacobianMatrix, i::Int, j::Int)
    if (i == j)
        return 2.0 * K.μ * norm((K.e[j].nodes[2] - K.e[j].nodes[1]) / 2.0) / (π * (norm(K.e[i].X - K.e[j].X)^2 - norm((K.e[j].nodes[2] - K.e[j].nodes[1]) / 2.0)^2)) + K.mat_loc[1,1][i]
    else
        return 2.0 * K.μ * norm((K.e[j].nodes[2] - K.e[j].nodes[1]) / 2.0) / (π * (norm(K.e[i].X - K.e[j].X)^2 - norm((K.e[j].nodes[2] - K.e[j].nodes[1]) / 2.0)^2))
    end
end

function Base.size(K::DD2DJacobianMatrix)
    return K.n, K.n
end

"""
Piecewise constant (PWC) three-dimensional axisymmetric jacobian matrix
"""
struct DD3DAxisymmetricJacobianMatrix{T<:Real} <: ElasticKernelMatrix{T}
    e::Vector{DDEdgeElem{T}}
    mat_loc::Matrix{Vector{T}}
    μ::T
    n::Int

    # Constructor
    function DD3DAxisymmetricJacobianMatrix(mesh::DDMesh1D{T}, mat_loc::Matrix{Vector{T}}, μ::T) where {T<:Real}
        return new{T}(mesh.elems, mat_loc, μ, length(mesh.elems))
    end
end

function Base.getindex(K::DD3DAxisymmetricJacobianMatrix, i::Int, j::Int)
    if (i == j)
        return K.μ / π * (((norm(K.e[j].nodes[2]) - norm(K.e[i].X)) * ellipe(m(K.e[i].X, K.e[j].nodes[1])) - (norm(K.e[j].nodes[1]) - norm(K.e[i].X)) * ellipe(m(K.e[i].X, K.e[j].nodes[2]))) / (norm(K.e[i].X - K.e[j].X)^2 - norm((K.e[j].nodes[2] - K.e[j].nodes[1]) / 2.0)^2) + ((norm(K.e[j].nodes[2]) + norm(K.e[i].X)) * ellipk(m(K.e[i].X, K.e[j].nodes[1])) - (norm(K.e[j].nodes[1]) + norm(K.e[i].X)) * ellipk(m(K.e[i].X, K.e[j].nodes[2]))) / (norm(K.e[i].X + K.e[j].X)^2 - norm((K.e[j].nodes[2] - K.e[j].nodes[1]) / 2.0)^2)) + K.mat_loc[1,1][i]
    else
        return K.μ / π * (((norm(K.e[j].nodes[2]) - norm(K.e[i].X)) * ellipe(m(K.e[i].X, K.e[j].nodes[1])) - (norm(K.e[j].nodes[1]) - norm(K.e[i].X)) * ellipe(m(K.e[i].X, K.e[j].nodes[2]))) / (norm(K.e[i].X - K.e[j].X)^2 - norm((K.e[j].nodes[2] - K.e[j].nodes[1]) / 2.0)^2) + ((norm(K.e[j].nodes[2]) + norm(K.e[i].X)) * ellipk(m(K.e[i].X, K.e[j].nodes[1])) - (norm(K.e[j].nodes[1]) + norm(K.e[i].X)) * ellipk(m(K.e[i].X, K.e[j].nodes[2]))) / (norm(K.e[i].X + K.e[j].X)^2 - norm((K.e[j].nodes[2] - K.e[j].nodes[1]) / 2.0)^2))
    end
end

function Base.size(K::DD3DAxisymmetricJacobianMatrix)
    return K.n, K.n
end

"""
Piecewise constant (PWC) three-dimensional normal jacobian matrix
"""
struct DD3DNormalJacobianMatrix{T<:Real} <: ElasticKernelMatrix{T}
    e::Vector{DDTriangleElem{T}}
    mat_loc::Matrix{Vector{T}}
    μ::T
    ν::T
    n::Int

    # Constructor
    function DD3DNormalJacobianMatrix(mesh::DDMesh2D{T}, mat_loc::Matrix{Vector{T}}, μ::T, ν::T) where {T<:Real}
        return new{T}(mesh.elems, mat_loc, μ, ν, length(mesh.elems))
    end
end

function Base.getindex(K::DD3DNormalJacobianMatrix, i::Int, j::Int)
    if (i == j)
        return K.μ / (4 * π * (1.0 - K.ν)) * integralI003(K.e[i], K.e[j]) + K.mat_loc[1,1][i]
    else
        return K.μ / (4 * π * (1.0 - K.ν)) * integralI003(K.e[i], K.e[j])
    end
end

function Base.size(K::DD3DNormalJacobianMatrix)
    return K.n, K.n
end

"""
Piecewise constant (PWC) three-dimensional shear axisymmetric jacobian matrix
"""
struct DD3DShearAxisymmetricJacobianMatrix{T<:Real} <: ElasticKernelMatrix{T}
    e::Vector{DDTriangleElem{T}}
    mat_loc::Matrix{Vector{T}}
    μ::T
    n::Int

    # Constructor
    function DD3DShearAxisymmetricJacobianMatrix(mesh::DDMesh2D{T}, mat_loc::Matrix{Vector{T}}, μ::T) where {T<:Real}
        return new{T}(mesh.elems, mat_loc, μ, length(mesh.elems))
    end
end

function Base.getindex(K::DD3DShearAxisymmetricJacobianMatrix, i::Int, j::Int)
    if (i == j)
        return K.μ / (4 * π) * integralI003(K.e[i], K.e[j]) + K.mat_loc[1,1][i]
    else
        return K.μ / (4 * π) * integralI003(K.e[i], K.e[j])
    end
end

function Base.size(K::DD3DShearAxisymmetricJacobianMatrix)
    return K.n, K.n
end

"""
Piecewise constant (PWC) three-dimensional shear jacobian matrix
"""
struct DD3DShearJacobianMatrix{T<:Real} <: ElasticKernelMatrix{T}
    e::Vector{DDTriangleElem{T}}
    mat_loc::Matrix{Vector{T}}
    μ::T
    ν::T
    n::Int

    # Constructor
    function DD3DShearJacobianMatrix(mesh::DDMesh2D{T}, mat_loc::Matrix{Vector{T}}, μ::T, ν::T) where {T<:Real}
        return new{T}(mesh.elems, mat_loc, μ, ν, length(mesh.elems))
    end
end

function Base.getindex(K::DD3DShearJacobianMatrix, i::Int, j::Int)
    if i <= K.n
        if j <= K.n
            if (i == j)
                return K.μ / (4 * π * (1.0 - K.ν)) * ((1.0 - 2 * K.ν) * integralI003(K.e[i], K.e[j]) + 3.0 * K.ν * integralI205(K.e[i], K.e[j])) + K.mat_loc[1,1][i]
            else
                return K.μ / (4 * π * (1.0 - K.ν)) * ((1.0 - 2 * K.ν) * integralI003(K.e[i], K.e[j]) + 3.0 * K.ν * integralI205(K.e[i], K.e[j]))
            end
        else
            if (i == (j - K.n))
                return 3.0 * K.ν * K.μ / (4 * π * (1.0 - K.ν)) * integralI115(K.e[i], K.e[j-K.n]) + K.mat_loc[1,2][i]
            else
                return 3.0 * K.ν * K.μ / (4 * π * (1.0 - K.ν)) * integralI115(K.e[i], K.e[j-K.n])
            end
        end
    else
        if j <= K.n
            if (i == (j + K.n))
                return 3.0 * K.ν * K.μ / (4 * π * (1.0 - K.ν)) * integralI115(K.e[i-K.n], K.e[j]) + K.mat_loc[2,1][i-K.n]
            else
                return 3.0 * K.ν * K.μ / (4 * π * (1.0 - K.ν)) * integralI115(K.e[i-K.n], K.e[j])
            end
        else
            if (i == j)
                return K.μ / (4 * π * (1.0 - K.ν)) * ((1.0 - 2 * K.ν) * integralI003(K.e[i-K.n], K.e[j-K.n]) + 3.0 * K.ν * integralI025(K.e[i-K.n], K.e[j-K.n])) + K.mat_loc[2,2][i-K.n]
            else
                return K.μ / (4 * π * (1.0 - K.ν)) * ((1.0 - 2 * K.ν) * integralI003(K.e[i-K.n], K.e[j-K.n]) + 3.0 * K.ν * integralI025(K.e[i-K.n], K.e[j-K.n]))
            end
        end
    end
end

function Base.size(K::DD3DShearJacobianMatrix)
    return 2 * K.n, 2 * K.n
end