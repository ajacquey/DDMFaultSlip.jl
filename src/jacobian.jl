import LinearAlgebra.mul!
import Base.*

# using HMatrices
abstract type AbstractDDJacobian{R,T<:Real} end

mutable struct NormalDDJacobian{R,T<:Real} <: AbstractDDJacobian{R,T}
    " Normal collocation matrix"
    En::HMatrix{R,T}

    " Local jacobian"
    jac_loc::Vector{T}

    " Constructor"
    function NormalDDJacobian(problem::NormalDDProblem{T}; eta::T=2.0, atol::T=1.0e-06) where {T<:Real}
        # Create H-matrix
        if isa(problem.mesh, DDMesh1D)
            K = DD2DElasticMatrix(problem.mesh.elems, problem.μ)
        elseif isa(problem.mesh, DDMesh2D)
            K = DD3DNormalElasticMatrix(problem.mesh.elems, problem.μ, problem.ν)
        end
        # Cluster tree
        Xclt = Yclt = ClusterTree([K.e[i].X for i in 1:length(K.e)])
        # Admissibility
        adm = StrongAdmissibilityStd(; eta=eta)
        # Compatibility
        comp = PartialACA(; atol=atol)
        # Assemble H-matrix
        En = assemble_hmat(K, Xclt, Yclt; adm, comp, threads=true, distributed=false)

        R = typeof(En.coltree)
        return new{R,T}(En, zeros(T, size(En, 1)))
    end
end

function Base.size(J::NormalDDJacobian{R,T}) where {R,T<:Real}
    return size(J.En)
end

function reinitLocalJacobian!(J::NormalDDJacobian{R,T}) where {R,T<:Real}
    fill!(J.jac_loc, 0.0)
end

mutable struct ShearDDJacobian2D{R,T<:Real} <: AbstractDDJacobian{R,T}
    " Shear collocation matrices"
    Es::HMatrix{R,T}

    " Local jacobian"
    jac_loc::Vector{T}

    " Constructor"
    function ShearDDJacobian2D(problem::ShearDDProblem2D{T}; eta::T=2.0, atol::T=1.0e-06) where {T<:Real}
        # Create H-matrix xx
        K = DD2DElasticMatrix(problem.mesh.elems, problem.μ)
        # Cluster tree
        Xclt = Yclt = ClusterTree([K.e[i].X for i in 1:length(K.e)])
        # Admissibility
        adm = StrongAdmissibilityStd(; eta=eta)
        # Compatibility
        comp = PartialACA(; atol=atol)
        # Assemble H-matrix
        Es = assemble_hmat(K, Xclt, Yclt; adm, comp, threads=true, distributed=false)

        R = typeof(Es.coltree)
        return new{R,T}(Es, zeros(T, size(Es, 1)))
    end
end

function Base.size(J::ShearDDJacobian2D{R,T}) where {R,T<:Real}
    return size(J.Es)
end

function reinitLocalJacobian!(J::ShearDDJacobian2D{R,T}) where {R,T<:Real}
    fill!(J.jac_loc, 0.0)
end

mutable struct ShearDDJacobian3D{R,T<:Real} <: AbstractDDJacobian{R,T}
    " Shear collocation matrices"
    Esxx::HMatrix{R,T}
    Esyy::HMatrix{R,T}
    Esxy::HMatrix{R,T}

    " Local jacobian"
    jac_loc_x::Vector{Vector{T}}
    jac_loc_y::Vector{Vector{T}}

    " Constructor"
    function ShearDDJacobian3D(problem::ShearDDProblem3D{T}; eta::T=2.0, atol::T=1.0e-06) where {T<:Real}
        # Create H-matrices
        Kxx = DD3DShearElasticMatrixXX(problem.mesh.elems, problem.μ, problem.ν)
        Kyy = DD3DShearElasticMatrixYY(problem.mesh.elems, problem.μ, problem.ν)
        Kxy = DD3DShearElasticMatrixXY(problem.mesh.elems, problem.μ, problem.ν)
        # Cluster tree
        Xclt = Yclt = ClusterTree([Kxx.e[i].X for i in 1:length(Kxx.e)])
        # Admissibility
        adm = StrongAdmissibilityStd(; eta=eta)
        # Compatibility
        comp = PartialACA(; atol=atol)
        # Assemble H-matrices
        Esxx = assemble_hmat(Kxx, Xclt, Yclt; adm, comp, threads=true, distributed=false)
        Esyy = assemble_hmat(Kyy, Xclt, Yclt; adm, comp, threads=true, distributed=false)
        Esxy = assemble_hmat(Kxy, Xclt, Yclt; adm, comp, threads=true, distributed=false)

        # Local jacobian
        jac_loc_x = Vector{Vector{T}}(undef, 2)
        jac_loc_x[1] = zeros(T, size(Esxx, 1))
        jac_loc_x[2] = zeros(T, size(Esxx, 1))
        jac_loc_y = Vector{Vector{T}}(undef, 2)
        jac_loc_y[1] = zeros(T, size(Esxx, 1))
        jac_loc_y[2] = zeros(T, size(Esxx, 1))

        R = typeof(Esxx.coltree)
        return new{R,T}(Esxx, Esyy, Esxy, jac_loc_x, jac_loc_y)
    end
end

function Base.size(J::ShearDDJacobian3D{R,T}) where {R,T<:Real}
    return size(J.Esxx, 1) + size(J.Esyy, 2), size(J.Esxx, 2) + size(J.Esyy, 2)
end

function reinitLocalJacobian!(J::ShearDDJacobian3D{R,T}) where {R,T<:Real}
    fill!(J.jac_loc_x[1], 0.0)
    fill!(J.jac_loc_x[2], 0.0)
    fill!(J.jac_loc_y[1], 0.0)
    fill!(J.jac_loc_y[2], 0.0)
end

mutable struct ShearNoNuDDJacobian3D{R,T<:Real} <: AbstractDDJacobian{R,T}
    " Shear collocation matrices"
    Esxx::HMatrix{R,T}
    Esyy::HMatrix{R,T}

    " Local jacobian"
    jac_loc_x::Vector{Vector{T}}
    jac_loc_y::Vector{Vector{T}}

    " Constructor"
    function ShearNoNuDDJacobian3D(problem::ShearDDProblem3D{T}; eta::T=2.0, atol::T=1.0e-06) where {T<:Real}
        # Create H-matrices
        Kxx = DD3DShearElasticMatrixXX(problem.mesh.elems, problem.μ, problem.ν)
        Kyy = DD3DShearElasticMatrixYY(problem.mesh.elems, problem.μ, problem.ν)
        # Cluster tree
        Xclt = Yclt = ClusterTree([Kxx.e[i].X for i in 1:length(Kxx.e)])
        # Admissibility
        adm = StrongAdmissibilityStd(; eta=eta)
        # Compatibility
        comp = PartialACA(; atol=atol)
        # Assemble H-matrices
        Esxx = assemble_hmat(Kxx, Xclt, Yclt; adm, comp, threads=true, distributed=false)
        Esyy = assemble_hmat(Kyy, Xclt, Yclt; adm, comp, threads=true, distributed=false)

        # Local jacobian
        jac_loc_x = Vector{Vector{T}}(undef, 2)
        jac_loc_x[1] = zeros(T, size(Esxx, 1))
        jac_loc_x[2] = zeros(T, size(Esxx, 1))
        jac_loc_y = Vector{Vector{T}}(undef, 2)
        jac_loc_y[1] = zeros(T, size(Esxx, 1))
        jac_loc_y[2] = zeros(T, size(Esxx, 1))

        R = typeof(Esxx.coltree)
        return new{R,T}(Esxx, Esyy, jac_loc_x, jac_loc_y)
    end
end

function Base.size(J::ShearNoNuDDJacobian3D{R,T}) where {R,T<:Real}
    return size(J.Esxx, 1) + size(J.Esyy, 2), size(J.Esxx, 2) + size(J.Esyy, 2)
end

function reinitLocalJacobian!(J::ShearNoNuDDJacobian3D{R,T}) where {R,T<:Real}
    fill!(J.jac_loc_x[1], 0.0)
    fill!(J.jac_loc_x[2], 0.0)
    fill!(J.jac_loc_y[1], 0.0)
    fill!(J.jac_loc_y[2], 0.0)
end

mutable struct CoupledDDJacobian2D{R,T<:Real} <: AbstractDDJacobian{R,T}
    " Collocation matrices"
    E::HMatrix{R,T}

    " Local jacobian"
    jac_loc_ϵ::Vector{Vector{T}}
    jac_loc_δ::Vector{Vector{T}}

    " Constructor"
    function CoupledDDJacobian2D(problem::CoupledDDProblem2D{T}; eta::T=2.0, atol::T=1.0e-06) where {T<:Real}
        # Create H-matrix xx
        K = DD2DElasticMatrix(problem.mesh.elems, problem.μ)
        # Cluster tree
        Xclt = Yclt = ClusterTree([K.e[i].X for i in 1:length(K.e)])
        # Admissibility
        adm = StrongAdmissibilityStd(; eta=eta)
        # Compatibility
        comp = PartialACA(; atol=atol)
        # Assemble H-matrix
        E = assemble_hmat(K, Xclt, Yclt; adm, comp, threads=true, distributed=false)

        # Local jacobian
        jac_loc_ϵ = Vector{Vector{T}}(undef, 2)
        jac_loc_ϵ[1] = zeros(T, size(E, 1))
        jac_loc_ϵ[2] = zeros(T, size(E, 1))
        jac_loc_δ = Vector{Vector{T}}(undef, 2)
        jac_loc_δ[1] = zeros(T, size(E, 1))
        jac_loc_δ[2] = zeros(T, size(E, 1))

        R = typeof(E.coltree)
        return new{R,T}(E, jac_loc_ϵ, jac_loc_δ)
    end
end

function Base.size(J::CoupledDDJacobian2D{R,T}) where {R,T<:Real}
    return 2 * size(J.E, 1), 2 * size(J.E, 2)
end

function reinitLocalJacobian!(J::CoupledDDJacobian2D{R,T}) where {R,T<:Real}
    fill!(J.jac_loc_ϵ[1], 0.0)
    fill!(J.jac_loc_ϵ[2], 0.0)
    fill!(J.jac_loc_δ[1], 0.0)
    fill!(J.jac_loc_δ[2], 0.0)
end

mutable struct CoupledDDJacobian3D{R,T<:Real} <: AbstractDDJacobian{R,T}
    " Normal collocation matrix"
    En::HMatrix{R,T}
    " Shear collocation matrices"
    Esxx::HMatrix{R,T}
    Esyy::HMatrix{R,T}
    Esxy::HMatrix{R,T}

    " Local jacobian"
    jac_loc_ϵ::Vector{Vector{T}}
    jac_loc_δx::Vector{Vector{T}}
    jac_loc_δy::Vector{Vector{T}}

    " Constructor"
    function CoupledDDJacobian3D(problem::CoupledDDProblem3D{T}; eta::T=2.0, atol::T=1.0e-06) where {T<:Real}
        # Create H-matrix
        Kn = DD3DNormalElasticMatrix(problem.mesh.elems, problem.μ, problem.ν)
        Ksxx = DD3DShearElasticMatrixXX(problem.mesh.elems, problem.μ, problem.ν)
        Ksyy = DD3DShearElasticMatrixYY(problem.mesh.elems, problem.μ, problem.ν)
        Ksxy = DD3DShearElasticMatrixXY(problem.mesh.elems, problem.μ, problem.ν)
        # Cluster tree
        Xclt = Yclt = ClusterTree([Kn.e[i].X for i in 1:length(Kn.e)])
        # Admissibility
        adm = StrongAdmissibilityStd(; eta=eta)
        # Compatibility
        comp = PartialACA(; atol=atol)
        # Assemble H-matrix
        En = assemble_hmat(Kn, Xclt, Yclt; adm, comp, threads=true, distributed=false)
        Esxx = assemble_hmat(Ksxx, Xclt, Yclt; adm, comp, threads=true, distributed=false)
        Esyy = assemble_hmat(Ksyy, Xclt, Yclt; adm, comp, threads=true, distributed=false)
        Esxy = assemble_hmat(Ksxy, Xclt, Yclt; adm, comp, threads=true, distributed=false)

        # Local jacobian
        jac_loc_ϵ = Vector{Vector{T}}(undef, 3)
        jac_loc_ϵ[1] = zeros(T, size(En, 1))
        jac_loc_ϵ[2] = zeros(T, size(En, 1))
        jac_loc_ϵ[3] = zeros(T, size(En, 1))
        jac_loc_δx = Vector{Vector{T}}(undef, 3)
        jac_loc_δx[1] = zeros(T, size(En, 1))
        jac_loc_δx[2] = zeros(T, size(En, 1))
        jac_loc_δx[3] = zeros(T, size(En, 1))
        jac_loc_δy = Vector{Vector{T}}(undef, 3)
        jac_loc_δy[1] = zeros(T, size(En, 1))
        jac_loc_δy[2] = zeros(T, size(En, 1))
        jac_loc_δy[3] = zeros(T, size(En, 1))

        R = typeof(En.coltree)
        return new{R,T}(En, Esxx, Esyy, Esxy, jac_loc_ϵ, jac_loc_δx, jac_loc_δy)
    end
end

function Base.size(J::CoupledDDJacobian3D{R,T}) where {R,T<:Real}
    return size(J.En, 1) + size(J.Esxx, 1) + size(J.Esyy, 1), size(J.En, 1) + size(J.Esxx, 1) + size(J.Esyy, 1)
end

function reinitLocalJacobian!(J::CoupledDDJacobian3D{R,T}) where {R,T<:Real}
    fill!(J.jac_loc_ϵ[1], 0.0)
    fill!(J.jac_loc_ϵ[2], 0.0)
    fill!(J.jac_loc_ϵ[3], 0.0)
    fill!(J.jac_loc_δx[1], 0.0)
    fill!(J.jac_loc_δx[2], 0.0)
    fill!(J.jac_loc_δx[3], 0.0)
    fill!(J.jac_loc_δy[1], 0.0)
    fill!(J.jac_loc_δy[2], 0.0)
    fill!(J.jac_loc_δy[3], 0.0)
end

mutable struct CoupledNoNuDDJacobian3D{R,T<:Real} <: AbstractDDJacobian{R,T}
    " Normal collocation matrix"
    En::HMatrix{R,T}
    " Shear collocation matrices"
    Esxx::HMatrix{R,T}
    Esyy::HMatrix{R,T}

    " Local jacobian"
    jac_loc_ϵ::Vector{Vector{T}}
    jac_loc_δx::Vector{Vector{T}}
    jac_loc_δy::Vector{Vector{T}}

    " Constructor"
    function CoupledNoNuDDJacobian3D(problem::CoupledDDProblem3D{T}; eta::T=2.0, atol::T=1.0e-06) where {T<:Real}
        # Create H-matrix
        Kn = DD3DNormalElasticMatrix(problem.mesh.elems, problem.μ, problem.ν)
        Ksxx = DD3DShearElasticMatrixXX(problem.mesh.elems, problem.μ, problem.ν)
        Ksyy = DD3DShearElasticMatrixYY(problem.mesh.elems, problem.μ, problem.ν)
        # Cluster tree
        Xclt = Yclt = ClusterTree([Kn.e[i].X for i in 1:length(Kn.e)])
        # Admissibility
        adm = StrongAdmissibilityStd(; eta=eta)
        # Compatibility
        comp = PartialACA(; atol=atol)
        # Assemble H-matrix
        En = assemble_hmat(Kn, Xclt, Yclt; adm, comp, threads=true, distributed=false)
        Esxx = assemble_hmat(Ksxx, Xclt, Yclt; adm, comp, threads=true, distributed=false)
        Esyy = assemble_hmat(Ksyy, Xclt, Yclt; adm, comp, threads=true, distributed=false)

        # Local jacobian
        jac_loc_ϵ = Vector{Vector{T}}(undef, 3)
        jac_loc_ϵ[1] = zeros(T, size(En, 1))
        jac_loc_ϵ[2] = zeros(T, size(En, 1))
        jac_loc_ϵ[3] = zeros(T, size(En, 1))
        jac_loc_δx = Vector{Vector{T}}(undef, 3)
        jac_loc_δx[1] = zeros(T, size(En, 1))
        jac_loc_δx[2] = zeros(T, size(En, 1))
        jac_loc_δx[3] = zeros(T, size(En, 1))
        jac_loc_δy = Vector{Vector{T}}(undef, 3)
        jac_loc_δy[1] = zeros(T, size(En, 1))
        jac_loc_δy[2] = zeros(T, size(En, 1))
        jac_loc_δy[3] = zeros(T, size(En, 1))

        R = typeof(En.coltree)
        return new{R,T}(En, Esxx, Esyy, jac_loc_ϵ, jac_loc_δx, jac_loc_δy)
    end
end

function Base.size(J::CoupledNoNuDDJacobian3D{R,T}) where {R,T<:Real}
    return size(J.En, 1) + size(J.Esxx, 1) + size(J.Esyy, 1), size(J.En, 1) + size(J.Esxx, 1) + size(J.Esyy, 1)
end

function reinitLocalJacobian!(J::CoupledNoNuDDJacobian3D{R,T}) where {R,T<:Real}
    fill!(J.jac_loc_ϵ[1], 0.0)
    fill!(J.jac_loc_ϵ[2], 0.0)
    fill!(J.jac_loc_ϵ[3], 0.0)
    fill!(J.jac_loc_δx[1], 0.0)
    fill!(J.jac_loc_δx[2], 0.0)
    fill!(J.jac_loc_δx[3], 0.0)
    fill!(J.jac_loc_δy[1], 0.0)
    fill!(J.jac_loc_δy[2], 0.0)
    fill!(J.jac_loc_δy[3], 0.0)
end

function Base.size(J::AbstractDDJacobian{R,T}, d::Integer) where {R,T<:Real}
    if d < 1 || d > 2
        throw(ArgumentError("dimension d must be ≥ 1 and ≤ 2, got $d"))
    else
        return size(J)[d]
    end
end

" Multiplication operator for AbstractDDJacobian"
function *(J::AbstractDDJacobian{R,T}, x::AbstractVector{T}) where {R,T<:Real}
    y = zeros(T, length(x))
    return mul!(y, J, x)
end

" mul! function for NormalDDJacobian"
function LinearAlgebra.mul!(y::AbstractVector{T}, J::NormalDDJacobian{R,T}, x::AbstractVector{T}, a::Number=1, b::Number=0;
    global_index=HMatrices.use_global_index(), threads=false) where {R,T<:Real}

    # Hmatrix multiplication
    collocation_mul!(y, J, x, a, b; global_index=global_index, threads=threads)

    # Add local jacobian contributions
    y .+= J.jac_loc .* x

    return y
end

" mul! function for ShearDDJacobian2D"
function LinearAlgebra.mul!(y::AbstractVector{T}, J::ShearDDJacobian2D{R,T}, x::AbstractVector{T}, a::Number=1, b::Number=0;
    global_index=HMatrices.use_global_index(), threads=false) where {R,T<:Real}

    # Hmatrix multiplication
    collocation_mul!(y, J, x, a, b; global_index=global_index, threads=threads)

    # Add local jacobian contributions
    y .+= J.jac_loc .* x

    return y
end

" mul! function for ShearNoNuDDJacobian3D"
function LinearAlgebra.mul!(y::AbstractVector{T}, J::ShearNoNuDDJacobian3D{R,T}, x::AbstractVector{T}, a::Number=1, b::Number=0;
    global_index=HMatrices.use_global_index(), threads=false) where {R,T<:Real}

    # Hmatrix multiplication
    collocation_mul!(y, J, x, a, b; global_index=global_index, threads=threads)

    # Add local jacobian contributions
    n = div(length(x), 2)
    y .+= vcat(J.jac_loc_x[1], J.jac_loc_y[2]) .* x
    y[1:n] .+= J.jac_loc_x[2] .* x[(n+1):(2*n)]
    y[(n+1):(2*n)] .+= J.jac_loc_y[1] .* x[1:n]

    return y
end


" mul! function for ShearDDJacobian3D"
function LinearAlgebra.mul!(y::AbstractVector{T}, J::ShearDDJacobian3D{R,T}, x::AbstractVector{T}, a::Number=1, b::Number=0;
    global_index=HMatrices.use_global_index(), threads=false) where {R,T<:Real}

    # Hmatrix multiplication
    collocation_mul!(y, J, x, a, b; global_index=global_index, threads=threads)

    # Add local jacobian contributions
    n = div(length(x), 2)
    y .+= vcat(J.jac_loc_x[1], J.jac_loc_y[2]) .* x
    y[1:n] .+= J.jac_loc_x[2] .* x[(n+1):(2*n)]
    y[(n+1):(2*n)] .+= J.jac_loc_y[1] .* x[1:n]

    return y
end

" mul! function for CoupledDDJacobian2D"
function LinearAlgebra.mul!(y::AbstractVector{T}, J::CoupledDDJacobian2D{R,T}, x::AbstractVector{T}, a::Number=1, b::Number=0;
    global_index=HMatrices.use_global_index(), threads=false) where {R,T<:Real}

    # Hmatrix multiplication
    collocation_mul!(y, J, x, a, b; global_index=global_index, threads=threads)

    # Add local jacobian contributions
    n = div(length(x), 2)
    y .+= vcat(J.jac_loc_ϵ[1], J.jac_loc_δ[2]) .* x
    y[1:n] .+= J.jac_loc_ϵ[2] .* x[(n+1):(2*n)]
    y[(n+1):(2*n)] .+= J.jac_loc_δ[1] .* x[1:n]

    return y
end

" mul! function for CoupledNoNuDDJacobian3D"
function LinearAlgebra.mul!(y::AbstractVector{T}, J::CoupledNoNuDDJacobian3D{R,T}, x::AbstractVector{T}, a::Number=1, b::Number=0;
    global_index=HMatrices.use_global_index(), threads=false) where {R,T<:Real}

    # Hmatrix multiplication
    collocation_mul!(y, J, x, a, b; global_index=global_index, threads=threads)

    # Add local jacobian contributions
    n = div(length(x), 3)
    y .+= vcat(J.jac_loc_ϵ[1], J.jac_loc_δx[2], J.jac_loc_δy[3]) .* x
    y[1:n] .+= J.jac_loc_ϵ[2] .* x[(n+1):(2*n)] .+ J.jac_loc_ϵ[3] .* x[(2*n+1):(3*n)]
    y[(n+1):(2*n)] .+= J.jac_loc_δx[1] .* x[1:n] .+ J.jac_loc_δx[3] .* x[(2*n+1):(3*n)]
    y[(2*n+1):(3*n)] .+= J.jac_loc_δy[1] .* x[1:n] .+ J.jac_loc_δy[2] .* x[(n+1):(2*n)]

    return y
end
" mul! function for CoupledNoNuDDJacobian2D"
function LinearAlgebra.mul!(y::AbstractVector{T}, J::CoupledDDJacobian3D{R,T}, x::AbstractVector{T}, a::Number=1, b::Number=0;
    global_index=HMatrices.use_global_index(), threads=false) where {R,T<:Real}

    # Hmatrix multiplication
    collocation_mul!(y, J, x, a, b; global_index=global_index, threads=threads)

    # Add local jacobian contributions
    n = div(length(x), 3)
    y .+= vcat(J.jac_loc_ϵ[1], J.jac_loc_δx[2], J.jac_loc_δy[3]) .* x
    y[1:n] .+= J.jac_loc_ϵ[2] .* x[(n+1):(2*n)] .+ J.jac_loc_ε[3] .* x[(2*n+1):(3*n)]
    y[(n+1):(2*n)] .+= J.jac_loc_δx[1] .* x[1:n] .+ J.jac_loc_δx[3] .* x[(2*n+1):(3*n)]
    y[(2*n+1):(3*n)] .+= J.jac_loc_δy[1] .* x[1:n] .+ J.jac_loc_δy[2] .* x[(n+1):(2*n)]

    return y
end

" collocation_mul! function for NormalDDJacobian"
function collocation_mul!(y::AbstractVector{T}, J::NormalDDJacobian{R,T}, x::AbstractVector{T}, a::Number=1, b::Number=0;
    global_index=HMatrices.use_global_index(), threads=false) where {R,T<:Real}
    mul!(y, J.En, x, a, b; global_index=global_index, threads=threads)
    return y
end

" collocation_mul! function for ShearDDJacobian2D"
function collocation_mul!(y::AbstractVector{T}, J::ShearDDJacobian2D{R,T}, x::AbstractVector{T}, a::Number=1, b::Number=0;
    global_index=HMatrices.use_global_index(), threads=false) where {R,T<:Real}
    mul!(y, J.Es, x, a, b; global_index=global_index, threads=threads)
    return y
end

" collocation_mul! function for ShearDDJacobian3D"
function collocation_mul!(y::AbstractVector{T}, J::ShearDDJacobian3D{R,T}, x::AbstractVector{T}, a::Number=1, b::Number=0;
    global_index=HMatrices.use_global_index(), threads=false) where {R,T<:Real}
    # Size of HMatrix
    n = size(J.Esxx, 1)

    # Slip δ_x multiplication
    mul!(view(y, 1:n), J.Esxx, view(x, 1:n), a, b; global_index=global_index, threads=threads)
    mul!(view(y, 1:n), J.Esxy, view(x, n+1:2*n), a, 1; global_index=global_index, threads=threads)
    # Slip δ_y multiplication
    mul!(view(y, n+1:2*n), J.Esyy, view(x, n+1:2*n), a, b; global_index=global_index, threads=threads)
    mul!(view(y, n+1:2*n), J.Esxy, view(x, 1:n), a, 1; global_index=global_index, threads=threads)

    return y
end

" collocation_mul! function for ShearNoNuDDJacobian3D"
function collocation_mul!(y::AbstractVector{T}, J::ShearNoNuDDJacobian3D{R,T}, x::AbstractVector{T}, a::Number=1, b::Number=0;
    global_index=HMatrices.use_global_index(), threads=false) where {R,T<:Real}
    # Size of HMatrix
    n = size(J.Esxx, 1)

    # Slip δ_x multiplication
    mul!(view(y, 1:n), J.Esxx, view(x, 1:n), a, b; global_index=global_index, threads=threads)
    # Slip δ_y multiplication
    mul!(view(y, n+1:2*n), J.Esyy, view(x, n+1:2*n), a, b; global_index=global_index, threads=threads)

    return y
end

" collocation_mul! function for CoupledDDJacobian2D"
function collocation_mul!(y::AbstractVector{T}, J::CoupledDDJacobian2D{R,T}, x::AbstractVector{T}, a::Number=1, b::Number=0;
    global_index=HMatrices.use_global_index(), threads=false) where {R,T<:Real}
    # Size of HMatrix
    n = size(J.E, 1)

    # Opening ϵ multiplication
    mul!(view(y, 1:n), J.E, view(x, 1:n), a, b; global_index=global_index, threads=threads)
    # Slip δ multiplication
    mul!(view(y, n+1:2*n), J.E, view(x, n+1:2*n), a, b; global_index=global_index, threads=threads)

    return y
end

" collocation_mul! function for CoupledDDJacobian3D"
function collocation_mul!(y::AbstractVector{T}, J::CoupledDDJacobian3D{R,T}, x::AbstractVector{T}) where {R,T<:Real}
    # Size of HMatrix
    n = size(J.En, 1)

    # Opening ϵ multiplication
    mul!(view(y, 1:n), J.En, view(x, 1:n), a, b; global_index=global_index, threads=threads)
    # Slip δ_x multiplication
    mul!(view(y, n+1:2*n), J.Esxx, view(x, n+1:2*n), a, b; global_index=global_index, threads=threads)
    mul!(view(y, n+1:2*n), J.Esxy, view(x, 2*n+1:3*n), a, 1; global_index=global_index, threads=threads)
    # Slip δ_y multiplication
    mul!(view(y, 2*n+1:3*n), J.Esyy, view(x, 2*n+1:3*n), a, b; global_index=global_index, threads=threads)
    mul!(view(y, 2*n+1:3*n), J.Esxy, view(x, n+1:2*n), a, 1; global_index=global_index, threads=threads)

    return y
end

" collocation_mul! function for CoupledNoNuDDJacobian3D"
function collocation_mul!(y::AbstractVector{T}, J::CoupledNoNuDDJacobian3D{R,T}, x::AbstractVector{T}, a::Number=1, b::Number=0;
    global_index=HMatrices.use_global_index(), threads=false) where {R,T<:Real}
    # Size of HMatrix
    n = size(J.En, 1)

    # Opening ϵ multiplication
    mul!(view(y, 1:n), J.En, view(x, 1:n), a, b; global_index=global_index, threads=threads)
    # Slip δ_x multiplication
    mul!(view(y, n+1:2*n), J.Esxx, view(x, n+1:2*n), a, b; global_index=global_index, threads=threads)
    # Slip δ_y multiplication
    mul!(view(y, 2*n+1:3*n), J.Esyy, view(x, 2*n+1:3*n), a, b; global_index=global_index, threads=threads)

    return y
end

" collocation_mul function for ShearDDJacobian3D"
function collocation_mul(J::ShearDDJacobian3D{R,T}, x::AbstractVector{T}, d::Integer;
    global_index=HMatrices.use_global_index(), threads=false) where {R,T<:Real}
    # Size of HMatrix
    n = size(J.Esxx, 1)
    # Declare results
    y = zeros(T, n)

    if (d == 1)
        # Slip δ_x multiplication
        mul!(y, J.Esxx, view(x, 1:n), 1, 0; global_index=global_index, threads=threads)
        mul!(y, J.Esxy, view(x, n+1:2*n), 1, 1; global_index=global_index, threads=threads)
        return y
    elseif (d == 2)
        # Slip δ_y multiplication
        mul!(y, J.Esyy, view(x, n+1:2*n), 1, 0; global_index=global_index, threads=threads)
        mul!(y, J.Esxy, view(x, 1:n), 1, 1; global_index=global_index, threads=threads)
        return y
    else
        throw(ErrorException("Wrong dimension!"))
    end
end

" collocation_mul function for ShearNoNuDDJacobian3D"
function collocation_mul(J::ShearNoNuDDJacobian3D{R,T}, x::AbstractVector{T}, d::Integer;
    global_index=HMatrices.use_global_index(), threads=false) where {R,T<:Real}
    # Size of HMatrix
    n = size(J.Esxx, 1)
    # Declare results
    y = zeros(T, n)

    if (d == 1)
        # Slip δ_x multiplication
        mul!(y, J.Esxx, view(x, 1:n), 1, 0; global_index=global_index, threads=threads)
        return y
    elseif (d == 2)
        # Slip δ_y multiplication
        mul!(y, J.Esyy, view(x, n+1:2*n), 1, 0; global_index=global_index, threads=threads)
        return y
    else
        throw(ErrorException("Wrong dimension!"))
    end
end

" collocation_mul function for CoupledDDJacobian2D"
function collocation_mul(J::CoupledDDJacobian2D{R,T}, x::AbstractVector{T}, d::Integer;
    global_index=HMatrices.use_global_index(), threads=false) where {R,T<:Real}
    # Size of HMatrix
    n = size(J.E, 1)
    # Declare results
    y = zeros(T, n)

    if (d == 0)
        # Opening ϵ multiplication
        mul!(y, J.E, view(x, 1:n), 1, 0; global_index=global_index, threads=threads)
        return y
    elseif (d == 1)
        # Slip δ multiplication
        mul!(y, J.E, view(x, n+1:2*n), 1, 0; global_index=global_index, threads=threads)
        return y
    else
        throw(ErrorException("Wrong dimension!"))
    end
end

" collocation_mul function for CoupledNoNuDDJacobian3D"
function collocation_mul(J::CoupledNoNuDDJacobian3D{R,T}, x::AbstractVector{T}, d::Integer;
    global_index=HMatrices.use_global_index(), threads=false) where {R,T<:Real}
    # Size of HMatrix
    n = size(J.En, 1)
    # Declare results
    y = zeros(T, n)

    if (d == 0)
        # Opening ϵ multiplication
        mul!(y, J.En, view(x, 1:n), 1, 0; global_index=global_index, threads=threads)
        return y
    elseif (d == 1)
        # Slip δ_x multiplication
        mul!(y, J.Esxx, view(x, n+1:2*n), 1, 0; global_index=global_index, threads=threads)
        return y
    elseif (d == 2)
        # Slip δ_y multiplication
        mul!(y, J.Esyy, view(x, 2*n+1:3*n), 1, 0; global_index=global_index, threads=threads)
        return y
    else
        throw(ErrorException("Wrong dimension!"))
    end
end

" collocation_mul function for CoupledDDJacobian3D"
function collocation_mul(J::CoupledDDJacobian3D{R,T}, x::AbstractVector{T}, d::Integer;
    global_index=HMatrices.use_global_index(), threads=false) where {R,T<:Real}
    # Size of HMatrix
    n = size(J.En, 1)
    # Declare results
    y = zeros(T, n)

    if (d == 0)
        # Opening ϵ multiplication
        mul!(y, J.En, view(x, 1:n), 1, 0; global_index=global_index, threads=threads)
        return y
    elseif (d == 1)
        # Slip δ_x multiplication
        mul!(y, J.Esxx, view(x, n+1:2*n), 1, 0; global_index=global_index, threads=threads)
        mul!(y, J.Esxy, view(x, 2*n+1:3*n), 1, 1; global_index=global_index, threads=threads)
        return y
    elseif (d == 2)
        # Slip δ_y multiplication
        mul!(y, J.Esyy, view(x, 2*n+1:3*n), 1, 0; global_index=global_index, threads=threads)
        mul!(y, J.Esxy, view(x, n+1:2*n), 1, 1; global_index=global_index, threads=threads)
        return y
    else
        throw(ErrorException("Wrong dimension!"))
    end
end