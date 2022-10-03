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
        
        # Local contributions - initialized to 0
        jac_loc = zeros(T, size(En, 1))

        R = typeof(En.coltree)
        return new{R,T}(En, jac_loc)
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
            
        # Local contributions - initialized to 0
        jac_loc = zeros(T, size(Es, 1))

        R = typeof(Es.coltree)
        return new{R,T}(Es, jac_loc)
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
        if (problem.ν != 0.0)
            Esxy = assemble_hmat(Kxy, Xclt, Yclt; adm, comp, threads=true, distributed=false)
        end

        R = typeof(Esxx.coltree)
        # return new{R,T}(Hmat, jac_loc)
        if (problem.ν != 0.0)
            return new{R,T}(Esxx, Esyy, Esxy, zeros(T, size(Esxx, 1)), zeros(T, size(Esxx, 1)))
        else
            return new{R,T}(Esxx, Esyy)
        end
    end
end

function Base.size(J::ShearDDJacobian3D{R,T}) where {R,T<:Real}
    return size(J.Esxx, 1) + size(J.Esyy, 2), size(J.Esxx, 2) + size(J.Esyy, 2)
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
    mul!(y, J.Es, -x, a, b; global_index=global_index, threads=threads)

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

# """
# Custom `show` function for `DDMJacobian{R,T}` that prints some information.
# """
# function Base.show(io::IO, J::DDMJacobian{R,T}) where {R,T<:Real}
#     println("Jacobian information:")
#     println("   -> H-matrix")
#     show(J.Hmat)
#     # println("   -> order: $(cps.order)")
#     # println("   -> n_cps: $(length(cps.cpoints))")
# end

# """
# Define `mul!` function for matrix vector multiplication with a DDMJacobian
# """
# function LinearAlgebra.mul!(y::AbstractVector{T}, J::DDMJacobian{R,T}, v::AbstractVector{T}) where {R,T<:Real}
#     # Number of variables
#     n_var = length(J.jac_loc)
#     # Size of H-mat
#     n = size(J.Hmat, 1)

#     # Loop over n_var
#     for i in 1:n_var
#         # H-matrix vector multiplication
#         y[(i - 1) * n + 1:i * n] = mul!(y[(i - 1) * n + 1:i * n], J.Hmat , v[(i - 1) * n + 1:i * n], 1, 0; threads=false, global_index=true)
#         # Local Jacobian multiplication
#         for j in 1:n_var
#             y[(i - 1) * n + 1:i * n] += J.jac_loc[i][(j - 1) * n + 1:j * n] .* v[(j - 1) * n + 1:j * n]
#         end
#     end

#     return y
# end

# """
# Define the `*` operator for matrix vector multiplication with a DDMJacobian
# """
# function LinearAlgebra.:*(J::DDMJacobian{R,T}, v::AbstractVector{T}) where {R,T<:Real}
#     return mul!(similar(v), J , v)
# end
