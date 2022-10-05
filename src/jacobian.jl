# using HMatrices
abstract type DDJacobian{R,T<:Real} end

mutable struct NormalDDJacobian{R,T<:Real} <: DDJacobian{R,T}
    " Normal collocation matrix"
    En::HMatrix{R,T}

    " Local jacobian"
    jac_loc::Vector{T}

    " Constructor"
    function NormalDDJacobian(problem::NormalDDProblem{T}; eta::T = 2.0, atol::T = 1.0e-06) where {T<:Real}
        # Create H-matrix
        if isa(problem.mesh, DDMesh1D)
            K = DD2DElasticMatrix(problem.mesh.elems, problem.μ)
        elseif isa(problem.mesh, DDMesh2D)
            K = DD3DNormalElasticMatrix(problem.mesh.elems, problem.μ, problem.ν)
        end
        # Cluster tree
        Xclt = Yclt = ClusterTree([K.e[i].X for i in 1:length(K.e)])
        # Admissibility
        adm = StrongAdmissibilityStd(; eta = eta)
        # Compatibility
        comp = PartialACA(; atol = atol)
        # Assemble H-matrix
        En = assemble_hmat(K, Xclt, Yclt; adm, comp, threads=true, distributed=false)

        R = typeof(En.coltree)
        return new{R,T}(En, zeros(T, size(En, 1)))
    end
end

function Base.size(J::NormalDDJacobian{R,T}) where {R,T<:Real}
    return size(J.En)
end

mutable struct ShearDDJacobian2D{R,T<:Real} <: DDJacobian{R,T}
    " Shear collocation matrices"
    Es::HMatrix{R,T}

    " Local jacobian"
    jac_loc::Vector{T}

    " Constructor"
    function ShearDDJacobian2D(problem::ShearDDProblem2D{T}; eta::T = 2.0, atol::T = 1.0e-06) where {T<:Real}
        # Create H-matrix xx
        K = DD2DElasticMatrix(problem.mesh.elems, problem.μ)
        # Cluster tree
        Xclt = Yclt = ClusterTree([K.e[i].X for i in 1:length(K.e)])
        # Admissibility
        adm = StrongAdmissibilityStd(; eta = eta)
        # Compatibility
        comp = PartialACA(; atol = atol)
        # Assemble H-matrix
        Es = assemble_hmat(K, Xclt, Yclt; adm, comp, threads=true, distributed=false)

        R = typeof(Es.coltree)
        return new{R,T}(Es, zeros(T, size(Es, 1)))
    end
end

function Base.size(J::ShearDDJacobian2D{R,T}) where {R,T<:Real}
    return size(J.Es)
end

mutable struct ShearDDJacobian3D{R,T<:Real} <: DDJacobian{R,T}
    " Shear collocation matrices"
    Esxx::HMatrix{R,T}
    Esyy::HMatrix{R,T}
    Esxy::HMatrix{R,T}

    " Local jacobian"
    jac_loc_x::Vector{T}
    jac_loc_y::Vector{T}

    " Constructor"
    function ShearDDJacobian3D(problem::ShearDDProblem3D{T}; eta::T = 2.0, atol::T = 1.0e-06) where {T<:Real}
        # Create H-matrices
        Kxx = DD3DShearElasticMatrixXX(problem.mesh.elems, problem.μ, problem.ν)
        Kyy = DD3DShearElasticMatrixYY(problem.mesh.elems, problem.μ, problem.ν)
        Kxy = DD3DShearElasticMatrixXY(problem.mesh.elems, problem.μ, problem.ν)
        # Cluster tree
        Xclt = Yclt = ClusterTree([Kxx.e[i].X for i in 1:length(Kxx.e)])
        # Admissibility
        adm = StrongAdmissibilityStd(; eta = eta)
        # Compatibility
        comp = PartialACA(; atol = atol)
        # Assemble H-matrices
        Esxx = assemble_hmat(Kxx, Xclt, Yclt; adm, comp, threads=true, distributed=false)
        Esyy = assemble_hmat(Kyy, Xclt, Yclt; adm, comp, threads=true, distributed=false)
        Esxy = assemble_hmat(Kxy, Xclt, Yclt; adm, comp, threads=true, distributed=false)

        R = typeof(Esxx.coltree)
        return new{R,T}(Esxx, Esyy, Esxy, zeros(T, size(Esxx, 1)), zeros(T, size(Esxx, 1)))
    end
end

function Base.size(J::ShearDDJacobian3D{R,T}) where {R,T<:Real}
    return size(J.Esxx, 1) + size(J.Esyy, 2), size(J.Esxx, 2) + size(J.Esyy, 2)
end

mutable struct ShearNoNuDDJacobian3D{R,T<:Real} <: DDJacobian{R,T}
    " Shear collocation matrices"
    Esxx::HMatrix{R,T}
    Esyy::HMatrix{R,T}

    " Local jacobian"
    jac_loc_x::Vector{T}
    jac_loc_y::Vector{T}

    " Constructor"
    function ShearNoNuDDJacobian3D(problem::ShearDDProblem3D{T}; eta::T = 2.0, atol::T = 1.0e-06) where {T<:Real}
        # Create H-matrices
        Kxx = DD3DShearElasticMatrixXX(problem.mesh.elems, problem.μ, problem.ν)
        Kyy = DD3DShearElasticMatrixYY(problem.mesh.elems, problem.μ, problem.ν)
        # Cluster tree
        Xclt = Yclt = ClusterTree([Kxx.e[i].X for i in 1:length(Kxx.e)])
        # Admissibility
        adm = StrongAdmissibilityStd(; eta = eta)
        # Compatibility
        comp = PartialACA(; atol = atol)
        # Assemble H-matrices
        Esxx = assemble_hmat(Kxx, Xclt, Yclt; adm, comp, threads=true, distributed=false)
        Esyy = assemble_hmat(Kyy, Xclt, Yclt; adm, comp, threads=true, distributed=false)

        R = typeof(Esxx.coltree)
        return new{R,T}(Esxx, Esyy, zeros(T, size(Esxx, 1)), zeros(T, size(Esxx, 1)))
    end
end

function Base.size(J::ShearNoNuDDJacobian3D{R,T}) where {R,T<:Real}
    return size(J.Esxx, 1) + size(J.Esyy, 2), size(J.Esxx, 2) + size(J.Esyy, 2)
end

mutable struct CoupledDDJacobian2D{R,T<:Real} <: DDJacobian{R,T}
    " Collocation matrices"
    E::HMatrix{R,T}

    " Local jacobian"
    jac_loc_ϵ::Vector{T}
    jac_loc_δ::Vector{T}

    " Constructor"
    function CoupledDDJacobian2D(problem::CoupledDDProblem2D{T}; eta::T = 2.0, atol::T = 1.0e-06) where {T<:Real}
        # Create H-matrix xx
        K = DD2DElasticMatrix(problem.mesh.elems, problem.μ)
        # Cluster tree
        Xclt = Yclt = ClusterTree([K.e[i].X for i in 1:length(K.e)])
        # Admissibility
        adm = StrongAdmissibilityStd(; eta = eta)
        # Compatibility
        comp = PartialACA(; atol = atol)
        # Assemble H-matrix
        E = assemble_hmat(K, Xclt, Yclt; adm, comp, threads=true, distributed=false)

        R = typeof(E.coltree)
        return new{R,T}(E, zeros(T, size(E, 1)), zeros(T, size(E, 1)))
    end
end

function Base.size(J::CoupledDDJacobian2D{R,T}) where {R,T<:Real}
    return 2 * size(J.E, 1), 2 * size(J.E, 2)
end

function Base.size(J::DDJacobian{R,T}, d::Integer) where {R,T<:Real}
    if d < 1 || d > 2
        throw(ArgumentError("dimension d must be ≥ 1 and ≤ 2, got $d"))
    else
        return size(J)[d]
    end
end

" mul! function for NormalDDJacobian"
function LinearAlgebra.mul!(y::AbstractVector, J::NormalDDJacobian{R,T}, x::AbstractVector, a::Number=1, b::Number=0;
    global_index=HMatrices.use_global_index(), threads=HMatrices.use_threads()) where {R,T<:Real}
    
    # Hmatrix multiplication
    mul!(y, J.En, x, a, b; global_index=global_index, threads=threads)

    # Add local jacobian contributions
    y .+= dot(J.jac_loc, x)

    return y
end

" mul! function for ShearDDJacobian2D"
function LinearAlgebra.mul!(y::AbstractVector, J::ShearDDJacobian2D{R,T}, x::AbstractVector, a::Number=1, b::Number=0;
    global_index=HMatrices.use_global_index(), threads=HMatrices.use_threads()) where {R,T<:Real}
    
    # Hmatrix multiplication
    collocation_mul!(y, J, x)

    # Add local jacobian contributions
    y .+= dot(J.jac_loc, x)

    return y
end

" mul! function for ShearDDJacobian3D"
function LinearAlgebra.mul!(y::AbstractVector, J::ShearDDJacobian3D{R,T}, x::AbstractVector, a::Number=1, b::Number=0;
    global_index=HMatrices.use_global_index(), threads=HMatrices.use_threads()) where {R,T<:Real}
    
    # Hmatrix multiplication
    collocation_mul!(y, J, x)

    # # Add local jacobian contributions
    # y .+= dot(J.jac_loc, x)

    return y
end

" mul! function for ShearNoNuDDJacobian3D"
function LinearAlgebra.mul!(y::AbstractVector, J::ShearNoNuDDJacobian3D{R,T}, x::AbstractVector, a::Number=1, b::Number=0;
    global_index=HMatrices.use_global_index(), threads=HMatrices.use_threads()) where {R,T<:Real}
    
    # Hmatrix multiplication
    collocation_mul!(y, J, x)

    # # Add local jacobian contributions
    # y .+= dot(J.jac_loc, x)

    return y
end

" mul! function for CoupledDDJacobian2D"
function LinearAlgebra.mul!(y::AbstractVector, J::CoupledDDJacobian2D{R,T}, x::AbstractVector, a::Number=1, b::Number=0;
    global_index=HMatrices.use_global_index(), threads=HMatrices.use_threads()) where {R,T<:Real}
    
    # Hmatrix multiplication
    collocation_mul!(y, J, x)

    # # Add local jacobian contributions
    # y .+= dot(J.jac_loc, x)

    return y
end

" collocation_mul! function for NormalDDJacobian"
function collocation_mul!(y::AbstractVector{T}, J::NormalDDJacobian{R,T}, x::AbstractVector{T}; global_index=HMatrices.use_global_index(), threads=HMatrices.use_threads()) where {R,T<:Real}
    mul!(y, J.En, x, 1, 0; global_index=global_index, threads=threads)
    return y
end

" collocation_mul! function for ShearDDJacobian2D"
function collocation_mul!(y::AbstractVector{T}, J::ShearDDJacobian2D{R,T}, x::AbstractVector{T}; global_index=HMatrices.use_global_index(), threads=HMatrices.use_threads()) where {R,T<:Real}
    mul!(y, J.Es, -x, 1, 0; global_index=global_index, threads=threads)
    return y
end

" collocation_mul! function for ShearDDJacobian3D"
function collocation_mul!(y::AbstractVector{T}, J::ShearDDJacobian3D{R,T}, x::AbstractVector{T}) where {R,T<:Real}
    n = size(J.Esxx, 1)

    y[1:n] = J.Esxx * x[1:n] + J.Esxy * x[n+1:2*n]
    y[n+1:2*n] = J.Esxy * x[1:n] + J.Esyy * x[n+1:2*n]
    return y
end

" collocation_mul! function for ShearNoNuDDJacobian3D"
function collocation_mul!(y::AbstractVector{T}, J::ShearNoNuDDJacobian3D{R,T}, x::AbstractVector{T}) where {R,T<:Real}
    n = size(J.Esxx, 1)

    y[1:n] = J.Esxx * x[1:n]
    y[n+1:2*n] = J.Esyy * x[n+1:2*n]

    return y
end

" collocation_mul! function for CoupledDDJacobian2D"
function collocation_mul!(y::AbstractVector{T}, J::CoupledDDJacobian2D{R,T}, x::AbstractVector{T}) where {R,T<:Real}
    n = size(J.E, 1)

    y[1:n] = J.E * x[1:n]
    y[n+1:2*n] = J.E * (-x[n+1:2*n])

    return y
end

" collocation_mul! function for ShearDDJacobian3D"
function collocation_mul(J::ShearDDJacobian3D{R,T}, x::AbstractVector{T}, d::Integer) where {R,T<:Real}
    n = size(J.Esxx, 1)

    if (d == 1)
        return J.Esxx * x[1:n] + J.Esxy * x[n+1:2*n]
    elseif (d == 2)
        return J.Esxy * x[1:n] + J.Esyy * x[n+1:2*n]
    else
        throw(ErrorException("Wrong dimension!"))
    end
end

" collocation_mul! function for ShearNoNuDDJacobian3D"
function collocation_mul(J::ShearNoNuDDJacobian3D{R,T}, x::AbstractVector{T}, d::Integer) where {R,T<:Real}
    n = size(J.Esxx, 1)

    if (d == 1)
        return J.Esxx * x[1:n]
    elseif (d == 2)
        return J.Esyy * x[n+1:2*n]
    else
        throw(ErrorException("Wrong dimension!"))
    end
end

" collocation_mul! function for CoupledDDJacobian2D"
function collocation_mul(J::CoupledDDJacobian2D{R,T}, x::AbstractVector{T}, d::Integer) where {R,T<:Real}
    n = size(J.E, 1)

    if (d == 0)
        return J.E * x[1:n]
    elseif (d == 1)
        return J.E * (-x[n+1:2*n])
    else
        throw(ErrorException("Wrong dimension!"))
    end
end