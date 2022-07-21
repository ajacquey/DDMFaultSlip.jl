using HMatrices

mutable struct DDMJacobian{R, T} <: AbstractMatrix{T}
    # Collocation matrix - HMatrix
    Hmat::HMatrix{R, T}
    # Jacobian form local kernels
    jac_loc::Vector{Vector{T}}
    # Constructor
    function DDMJacobian(K::ElasticKernelMatrix{T}, n_var::Int;eta::T=3.0, atol::T=0.0) where {T<:Real}
        # Create H-matrix
        # Cluster tree
        Xclt = Yclt = ClusterTree(K.cpi.cpoints)
        # Admissibility
        adm = StrongAdmissibilityStd(;eta=eta)
        # Compatibility
        comp = PartialACA(;atol=atol)
        # Assemble H-matrix
        Hmat = assemble_hmat(K, Xclt, Yclt;adm, comp, threads=true, distributed=false)
        
        # Local contributions - initialized to 0
        jac_loc = Vector{Vector{T}}(undef, n_var)
        for i in 1:n_var
            jac_loc[i] = zeros(T, n_var * size(K, 1))
        end

        R = typeof(Hmat.coltree)
        return new{R,T}(Hmat, jac_loc)
    end
end

function Base.size(J::DDMJacobian{R,T}) where {R,T<:Real}
    # Size = size of H-matrix times number of variables
    return length(J.jac_loc) .* size(J.Hmat)
end

function Base.getindex(J::DDMJacobian{R,T}, i::Int, j::Int) where {R,T<:Real}
    # Number of variables
    n_var = length(J.jac_loc)
    # Size of H-mat
    n = size(J.Hmat, 1)
    if n_var == 1
        if i != j
            return getindex(J.Hmat, i, j)
        else
            return getindex(J.Hmat, i, j) + J.jac_loc[1][i]
        end
    else
        # Inside H-matrix
        if div(i - 1, n) == div(j - 1, n)
            if i != j
                return getindex(J.Hmat, i - n * div(i - 1, n), j - n* div(j - 1, n))
            else # Diagonal
                return getindex(J.Hmat, i - n * div(i - 1, n), j - n* div(j - 1, n)) + J.jac_loc[div(i - 1, n) + 1][i]
            end
        else # Outside H-matrix
            if rem(abs(i - j), n) != 0
                return 0.0
            else
                return J.jac_loc[div(i - 1, n) + 1][j]
            end
        end
    end
end

"""
Custom `show` function for `DDMJacobian{R,T}` that prints some information.
"""
function Base.show(io::IO, J::DDMJacobian{R,T}) where {R,T<:Real}
    println("Jacobian information:")
    println("   -> H-matrix")
    show(J.Hmat)
    # println("   -> order: $(cps.order)")
    # println("   -> n_cps: $(length(cps.cpoints))")
end

"""
Define `mul!` function for matrix vector multiplication with a DDMJacobian
"""
function LinearAlgebra.mul!(y::AbstractVector{T}, J::DDMJacobian{R,T}, v::AbstractVector{T}) where {R,T<:Real}
    # Number of variables
    n_var = length(J.jac_loc)
    # Size of H-mat
    n = size(J.Hmat, 1)

    # Loop over n_var
    for i in 1:n_var
        # H-matrix vector multiplication
        y[(i - 1) * n + 1:i * n] = mul!(y[(i - 1) * n + 1:i * n], J.Hmat , v[(i - 1) * n + 1:i * n], 1, 0; threads=false, global_index=true)
        # Local Jacobian multiplication
        for j in 1:n_var
            y[(i - 1) * n + 1:i * n] += J.jac_loc[i][(j - 1) * n + 1:j * n] .* v[(j - 1) * n + 1:j * n]
        end
    end

    return y
end

"""
Define the `*` operator for matrix vector multiplication with a DDMJacobian
"""
function LinearAlgebra.:*(J::DDMJacobian{R,T}, v::AbstractVector{T}) where {R,T<:Real}
    return mul!(similar(v), J , v)
end
