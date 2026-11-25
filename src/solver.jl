import LinearAlgebra.mul!

"""
Jacobian operator J = E + S, where E is the (fixed) collocation H-matrix and S is
the sparse local jacobian assembled from the constraints. Only S changes between
nonlinear iterations, so the H-matrix is never reassembled.
"""
mutable struct DDJacobianOperator{R,T<:Real}
    " The collocation matrix"
    E::HMatrix{R,T}

    " The sparse local jacobian"
    S::SparseMatrixCSC{T,Int}
end

Base.size(J::DDJacobianOperator) = size(J.E)
Base.size(J::DDJacobianOperator, d::Integer) = size(J.E, d)
Base.eltype(::DDJacobianOperator{R,T}) where {R,T<:Real} = T

function LinearAlgebra.mul!(y::AbstractVector{T}, J::DDJacobianOperator{R,T}, x::AbstractVector{T}) where {R,T<:Real}
    mul!(y, J.E, x)
    mul!(y, J.S, x, true, true)
    return y
end

Base.:*(J::DDJacobianOperator{R,T}, x::AbstractVector{T}) where {R,T<:Real} = mul!(similar(x), J, x)

" Sparse matrix from the local jacobian blocks"
function localJacobianMatrix(mat_loc::Matrix{Vector{T}}) where {T<:Real}
    if size(mat_loc) == (1, 1)
        return spdiagm(0 => mat_loc[1, 1])
    else
        return [spdiagm(0 => mat_loc[1, 1]) spdiagm(0 => mat_loc[1, 2]); spdiagm(0 => mat_loc[2, 1]) spdiagm(0 => mat_loc[2, 2])]
    end
end

"""
    DDSolver

Nonlinear solver of a DD problem, created by [`run!`](@ref). The elastic
collocation matrix is assembled once as a hierarchical matrix, and the nonlinear
system is solved with Newton iterations, using a preconditioned Krylov method for
the linear systems.
"""
mutable struct DDSolver{R,T<:Real}
    " The jacobian operator"
    mat::DDJacobianOperator{R,T}

    " The local jacobian matrix"
    mat_loc::Matrix{Vector{T}}
    
    " The collocation matrix"
    E::HMatrix{R,T}

    " The elastic matrix"
    K::ElasticKernelMatrix{T}

    " The H-matrix compressor"
    comp::PartialACA

    " The H-matrix admissibility"
    adm::StrongAdmissibilityStd

    " The H-matrix cluster"
    Xclt::R

    " Preconditioner"
    Pc::Union{LU{T,HMatrix{R,T}},Identity}

    " The residual vector"
    rhs::Vector{T}

    " The solution vector"
    solution::Vector{T}

    " Preconditioner"
    pc::Bool

    " Preconditioner tolerance"
    pc_atol::T

    " Recompute the preconditioner when the last linear solve exceeded this number of iterations"
    pc_refresh_it::Int

    " Recompute the preconditioner when the true relative residual of the last linear solve exceeded this tolerance"
    pc_lin_rtol::T

    " Local jacobian used to build the current preconditioner"
    pc_S::SparseMatrixCSC{T,Int}

    " Number of iterations of the last linear solve"
    l_iters::Int

    " True relative residual of the last linear solve"
    l_res::T

    " Maximum number of nonlinear iterations"
    nl_max_it::Int

    " Nonlinear absolute tolerance"
    nl_abs_tol::T

    " Nonlinear relative tolerance"
    nl_rel_tol::T

    " Linear solver"
    l_solver::String

    " Maximum number of linear iterations"
    l_max_it::Int

    " Linear absolute tolerance"
    l_abs_tol::T

    " Linear relative tolerance"
    l_rel_tol::T

    " Check if linear solver is supportted"
    function checkLinearSolver(l_solver)
        # Check if linear solver is provided
        supported_linear_solver = ["bicgstabl", "gmres", "idrs"]
        if ~(l_solver in supported_linear_solver)
            throw(ErrorException("The linear solver $(l_solver) is not supported!"))
        end
        return nothing
    end

    " Constructor for NormalDDProblem"
    function DDSolver(problem::NormalDDProblem{T};
        hmat_eta::T=3.0, hmat_atol::T=1.0e-08,
        pc::Bool=true, pc_atol::T=1.0e-02, pc_refresh_it::Int=10, pc_lin_rtol::T=1.0e-06,
        nl_abs_tol::T, nl_rel_tol::T, nl_max_it::Int,
        l_solver::String, l_max_it::Int, l_abs_tol::T, l_rel_tol::T) where {T<:Real}

        # Local jacobian
        mat_loc = Matrix{Vector{T}}(undef, 1, 1)
        for i in eachindex(mat_loc)
            mat_loc[i] = zeros(problem.n)
        end

        # Assemble collocation matrix
        if isa(problem.mesh, DDMesh1D)
            if (problem.axisymmetric)
                K = DD3DAxisymmetricElasticMatrix(problem.mesh, problem.μ)
                Kj = DD3DAxisymmetricJacobianMatrix(problem.mesh, mat_loc, problem.μ)
            else
                K = DD2DElasticMatrix(problem.mesh, problem.μ)
                Kj = DD2DJacobianMatrix(problem.mesh, mat_loc, problem.μ)
            end
        elseif isa(problem.mesh, DDMesh2D)
            K = DD3DNormalElasticMatrix(problem.mesh, problem.μ, problem.ν)
            Kj = DD3DNormalJacobianMatrix(problem.mesh, mat_loc, problem.μ, problem.ν)
        end
        # Cluster tree
        Xclt = ClusterTree([problem.mesh.elems[i].X for i in eachindex(problem.mesh.elems)])
        # Admissibility
        adm = StrongAdmissibilityStd(; eta=hmat_eta)
        # Compatibility
        comp = PartialACA(; atol=hmat_atol)
        # Assemble H-matrix
        E = assemble_hmatrix(K, Xclt, Xclt; adm, comp, global_index=true, threads=false, distributed=false)
        mat = DDJacobianOperator(E, localJacobianMatrix(mat_loc))

        # Preconditioner
        Pc = IterativeSolvers.Identity()

        # Degrees of freedom
        n_dof = size(mat, 1)

        # Check if linear solver is provided
        checkLinearSolver(l_solver)

        R = typeof(E.coltree)
        return new{R,T}(mat, mat_loc, E, Kj, comp, adm, Xclt, 
            Pc, zeros(T, n_dof), zeros(T, n_dof),
            pc, pc_atol, pc_refresh_it, pc_lin_rtol, mat.S, 0, zero(T),
            nl_max_it, nl_abs_tol, nl_rel_tol,
            l_solver, l_max_it, l_abs_tol, l_rel_tol)
    end

    " Constructor for ShearDDProblem"
    function DDSolver(problem::ShearDDProblem{T};
        hmat_eta::T=3.0, hmat_atol::T=1.0e-08,
        pc::Bool=true, pc_atol::T=1.0e-02, pc_refresh_it::Int=10, pc_lin_rtol::T=1.0e-06,
        nl_abs_tol::T, nl_rel_tol::T, nl_max_it::Int,
        l_solver::String, l_max_it::Int, l_abs_tol::T, l_rel_tol::T) where {T<:Real}

        # Local jacobian
        if (isa(problem.mesh, DDMesh1D) || (isa(problem.mesh, DDMesh2D) && (problem.ν == 0.0)))
            mat_loc = Matrix{Vector{T}}(undef, 1, 1)
        else
            mat_loc = Matrix{Vector{T}}(undef, 2, 2)
        end
        for i in eachindex(mat_loc)
            mat_loc[i] = zeros(problem.n)
        end

        # Assemble collocation matrix
        if isa(problem.mesh, DDMesh1D)
            if (problem.axisymmetric)
                K = DD3DAxisymmetricElasticMatrix(problem.mesh, problem.μ)
                Kj = DD3DAxisymmetricJacobianMatrix(problem.mesh, mat_loc, problem.μ)
            else
                K = DD2DElasticMatrix(problem.mesh, problem.μ)
                Kj = DD2DJacobianMatrix(problem.mesh, mat_loc, problem.μ)
            end
            # Cluster tree
            Xclt = ClusterTree([problem.mesh.elems[i].X for i in eachindex(problem.mesh.elems)])
        else
            if (problem.ν == 0.0)
                K = DD3DShearAxisymmetricElasticMatrix(problem.mesh, problem.μ)
                Kj = DD3DShearAxisymmetricJacobianMatrix(problem.mesh, mat_loc, problem.μ)
                Xclt = ClusterTree([problem.mesh.elems[i].X for i in eachindex(problem.mesh.elems)])
            else
                K = DD3DShearElasticMatrix(problem.mesh, problem.μ, problem.ν)
                Kj = DD3DShearJacobianMatrix(problem.mesh, mat_loc, problem.μ, problem.ν)
                # Cluster tree
                Xclt = ClusterTree(repeat([problem.mesh.elems[i].X for i in eachindex(problem.mesh.elems)], 2))
            end
        end
        # Admissibility
        adm = StrongAdmissibilityStd(; eta=hmat_eta)
        # Compatibility
        comp = PartialACA(; atol=hmat_atol)
        # Assemble H-matrix
        E = assemble_hmatrix(K, Xclt, Xclt; adm, comp, global_index=true, threads=true, distributed=false)
        mat = DDJacobianOperator(E, localJacobianMatrix(mat_loc))

        # Preconditioner
        Pc = IterativeSolvers.Identity()

        # Degrees of freedom
        n_dof = size(E, 1)

        # Check if linear solver is provided
        checkLinearSolver(l_solver)

        R = typeof(E.coltree)
        return new{R,T}(mat, mat_loc, E, Kj, comp, adm, Xclt, 
            Pc, zeros(T, n_dof), zeros(T, n_dof),
            pc, pc_atol, pc_refresh_it, pc_lin_rtol, mat.S, 0, zero(T),
            nl_max_it, nl_abs_tol, nl_rel_tol,
            l_solver, l_max_it, l_abs_tol, l_rel_tol)
    end
end

function linear_solve!(dx::Vector{T}, solver::DDSolver{R,T}, log::Bool) where {R,T<:Real}
    # Permutation
    if ~HMatrices.use_global_index()
        cperm = HMatrices.colperm(solver.E) # column permutation
        rperm = HMatrices.rowperm(solver.E) # row permutation
        solver.rhs = solver.rhs[cperm]
    end

    if solver.l_solver == "bicgstabl"
        dx, ch = bicgstabl!(dx, solver.mat, -solver.rhs; Pl=solver.Pc, log=true, abstol=solver.l_abs_tol, reltol=solver.l_rel_tol, max_mv_products=4*solver.l_max_it)
    elseif solver.l_solver == "gmres"
        dx, ch = gmres!(dx, solver.mat, -solver.rhs; Pl=solver.Pc, log=true, abstol=solver.l_abs_tol, reltol=solver.l_rel_tol, maxiter=solver.l_max_it)
    elseif solver.l_solver == "idrs"
        dx, ch = idrs!(dx, solver.mat, -solver.rhs; Pl=solver.Pc, log=true, abstol=solver.l_abs_tol, reltol=solver.l_rel_tol, maxiter=solver.l_max_it)
    end

    # Convergence of the (left) preconditioned solver is checked on the preconditioned
    # residual: also compute the true residual to detect an outdated preconditioner
    solver.l_iters = ch.iters
    solver.l_res = norm(solver.mat * dx + solver.rhs) / norm(solver.rhs)

    if log
        if ch.isconverged
            @printf("    -> Linear Solve converged after %i iterations (|J dx + R| / |R| = %e).\n", ch.iters, solver.l_res)
        else
            @printf("    -> Linear Solve did NOT converge after %i iterations (|J dx + R| / |R| = %e).\n", ch.iters, solver.l_res)
        end
    end

    # Inverse permutation
    if ~HMatrices.use_global_index()
        invpermute!(dx,rperm)
    end

    return dx
end

function print_NL_res(it::Int, r::T) where {T<:Real}
    @printf("  %i Nonlinear |R| = %e\n", it, r)
    return nothing
end

" Hierarchical LU of the jacobian J = E + S used as preconditioner"
function updatePreconditioner!(solver::DDSolver{R,T}) where {R,T<:Real}
    M = deepcopy(solver.E)
    axpy!(one(T), solver.mat.S, M)
    solver.Pc = lu!(M; atol=solver.pc_atol)
    solver.pc_S = solver.mat.S
    return nothing
end

" Solve the problem using the IterativeSolvers package"
function solve!(solver::DDSolver{R,T}, problem::AbstractDDProblem{T}, dt::T, timer::TimerOutput; log::Bool=true, linear_log::Bool=false)::Bool where {R,T<:Real}
    ##### Newton loop #####
    # Non-linear iterations
    nl_iter = 0
    # Declare solution
    dx = zeros(T, length(solver.rhs))
    # Initial residual
    assembleResidualAndJacobian!(solver, problem, dt, timer)
    r = norm(solver.rhs)
    r0 = r
    if log
        print_NL_res(0, r0)
    end
    # Main loop
    while (nl_iter <= solver.nl_max_it)
        # Preconditioner: reused across nonlinear iterations and time steps and
        # only recomputed when it degrades (the jacobian itself is always exact)
        if (solver.pc && (isa(solver.Pc, Identity) || (solver.l_iters > solver.pc_refresh_it)))
            @timeit timer "Preconditionning" updatePreconditioner!(solver)
        end
        # Linear Solve
        @timeit timer "Linear solve" dx = linear_solve!(dx, solver, linear_log)
        # Outdated preconditioner: recompute it and solve again
        if (solver.pc && (solver.l_res > solver.pc_lin_rtol) && (solver.pc_S !== solver.mat.S))
            @timeit timer "Preconditionning" updatePreconditioner!(solver)
            fill!(dx, 0.0)
            @timeit timer "Linear solve" dx = linear_solve!(dx, solver, linear_log)
        end

        # Update solution
        solver.solution .+= dx
        # Update problem
        @timeit timer "Update problem" update!(problem, solver)
        # Update residuals and jacobian
        assembleResidualAndJacobian!(solver, problem, dt, timer)
        r = norm(solver.rhs)

        nl_iter += 1
        if log
            print_NL_res(nl_iter, norm(r))
        end

        # Check convergence
        if (r <= solver.nl_abs_tol)
            if log
                @printf("Solve converged with absolute tolerance!\n")
            end
            return true
        end
        if (r / r0 <= solver.nl_rel_tol)
            if log
                @printf("Solve converged with relative tolerance!\n")
            end
            return true
        end
    end
    # Error if exceeded maximum number of iterations
    if (nl_iter > solver.nl_max_it) 
        if log
            @printf("Solve diverged: exceeded the maximum number of nonlinear iterations!\n")
        end
        return false
    end
end

" Overwrite mul! function for HMatrix{R,T}"
function LinearAlgebra.mul!(y::AbstractVector{T}, A::HMatrix{R,T}, x::AbstractVector{T}, a::Number=1, b::Number=0) where {R,T<:Real}

    HMatrices.mul!(y, A ,x, a, b; global_index=HMatrices.use_global_index(), threads=HMatrices.use_threads())

    return y
end

function Base.show(io::IO, solver::DDSolver{R,T}) where {R,T<:Real}
    # Nonlinear system
    @printf(io, "Nonlinear system:\n")
    @printf(io, "  Num DoFs: %i\n", size(solver.E, 1))
    @printf(io, "  Use threads with HMatrix: ")
    @printf(io, "%s", HMatrices.use_threads() ? "true\n" : "false\n")
    @printf(io, "  Use global index with HMatrix: ")
    @printf(io, "%s", HMatrices.use_global_index() ? "true\n" : "false\n")
    @printf(io, "  Linear solver: ")
    @printf(io, "%s\n", solver.l_solver)
    @printf(io, "  Preconditioner: ")
    @printf(io, "%s", solver.pc ? "ILU\n\n" : "none\n\n")
end

function reinitLocalJacobian!(mat_loc::Matrix{Vector{T}}) where {T<:Real}
    for i in eachindex(mat_loc)
        fill!(mat_loc[i], 0.0)
    end
    return nothing
end

" Reinitialize solver after successful solve"
function reinit!(solver::DDSolver{R,T}; end_time_step::Bool=false) where {R,T<:Real}
    fill!(solver.rhs, 0.0)
    reinitLocalJacobian!(solver.mat_loc)
    if end_time_step
        fill!(solver.solution, 0.0)
    end
    return nothing
end