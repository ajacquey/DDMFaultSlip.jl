import LinearAlgebra.mul!
mutable struct DDSolver{R,T<:Real}
    " The jacobian matrix"
    mat::HMatrix{R,T}

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
        pc::Bool=true, pc_atol::T=1.0e-02,
        nl_abs_tol::T, nl_rel_tol::T, nl_max_it::Int,
        l_solver::String, l_max_it::Int, l_abs_tol::T, l_rel_tol::T) where {T<:Real}

        # Local jacobian
        mat_loc = Matrix{Vector{T}}(undef, 1, 1)
        for i in eachindex(mat_loc)
            mat_loc[i] = zeros(problem.n)
        end

        # Assemble collocation matrix
        if isa(problem.mesh, DDMesh1D)
            K = DD2DElasticMatrix(problem.mesh, problem.μ)
            Kj = DD2DJacobianMatrix(problem.mesh, mat_loc, problem.μ)
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
        E = assemble_hmatrix(K, Xclt, Xclt; adm, comp, global_index=true, threads=true, distributed=false)
        mat = E

        # Preconditioner
        Pc = IterativeSolvers.Identity()

        # Degrees of freedom
        n_dof = size(mat, 1)

        # Check if linear solver is provided
        checkLinearSolver(l_solver)

        R = typeof(E.coltree)
        return new{R,T}(mat, mat_loc, E, Kj, comp, adm, Xclt, 
            Pc, zeros(T, n_dof), zeros(T, n_dof),
            pc, pc_atol,
            nl_max_it, nl_abs_tol, nl_rel_tol,
            l_solver, l_max_it, l_abs_tol, l_rel_tol)
    end

    " Constructor for ShearDDProblem"
    function DDSolver(problem::ShearDDProblem{T};
        hmat_eta::T=3.0, hmat_atol::T=1.0e-08,
        pc::Bool=true, pc_atol::T=1.0e-02,
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
            K = DD2DElasticMatrix(problem.mesh, problem.μ)
            Kj = DD2DJacobianMatrix(problem.mesh, mat_loc, problem.μ)
            # Cluster tree
            Xclt = ClusterTree([problem.mesh.elems[i].X for i in eachindex(problem.mesh.elems)])
        else
            if (problem.ν == 0.0)
                K = DD3DShearAxisSymmetricElasticMatrix(problem.mesh, problem.μ)
                Kj = DD3DShearAxisSymmetricJacobianMatrix(problem.mesh, mat_loc, problem.μ)
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
        mat = E

        # Preconditioner
        Pc = IterativeSolvers.Identity()

        # Degrees of freedom
        n_dof = size(E, 1)

        # Check if linear solver is provided
        checkLinearSolver(l_solver)

        R = typeof(E.coltree)
        return new{R,T}(mat, mat_loc, E, Kj, comp, adm, Xclt, 
            Pc, zeros(T, n_dof), zeros(T, n_dof),
            pc, pc_atol,
            nl_max_it, nl_abs_tol, nl_rel_tol,
            l_solver, l_max_it, l_abs_tol, l_rel_tol)
    end
end

function linear_solve!(dx::Vector{T}, solver::DDSolver{R,T}, log::Bool) where {R,T<:Real}
    # Permutation
    if ~HMatrices.use_global_index()
        cperm = HMatrices.colperm(solver.mat) # column permutation
        rperm = HMatrices.rowperm(solver.mat) # row permutation
        solver.rhs = solver.rhs[cperm]
    end

    if solver.l_solver == "bicgstabl"
        dx, ch = bicgstabl!(dx, solver.mat, -solver.rhs; Pl=solver.Pc, log=true, abstol=solver.l_abs_tol, reltol=solver.l_rel_tol, max_mv_products=4*solver.l_max_it)
    elseif solver.l_solver == "gmres"
        dx, ch = gmres!(dx, solver.mat, -solver.rhs; Pl=solver.Pc, log=true, abstol=solver.l_abs_tol, reltol=solver.l_rel_tol, maxiter=solver.l_max_it)
    elseif solver.l_solver == "idrs"
        dx, ch = idrs!(dx, solver.mat, -solver.rhs; Pl=solver.Pc, log=true, abstol=solver.l_abs_tol, reltol=solver.l_rel_tol, maxiter=solver.l_max_it)
    end

    if log
        if ch.isconverged
            @printf("    -> Linear Solve converged after %i iterations.\n", ch.iters)
        else
            @printf("    -> Linear Solve did NOT converge after %i iterations.\n", ch.iters)
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
    # Preconditioner
    if (solver.pc)
        @timeit timer "Preconditionning" solver.Pc = lu(solver.mat; atol=solver.pc_atol)
    end
    if log
        print_NL_res(0, r0)
    end
    # Main loop
    while (nl_iter <= solver.nl_max_it)
        # Linear Solve
        @timeit timer "Linear solve" dx = linear_solve!(dx, solver, linear_log)

        # Update solution
        solver.solution .+= dx
        # Update problem
        @timeit timer "Update problem" update!(problem, solver)
        # Update residuals and jacobian
        assembleResidualAndJacobian!(solver, problem, dt, timer)
        r = norm(solver.rhs)
        # Preconditioner
        if (solver.pc)
            @timeit timer "Preconditionning" solver.Pc = lu(solver.mat; atol=solver.pc_atol)
        end

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
    @printf("Nonlinear system:\n")
    @printf("  Num DoFs: %i\n", size(solver.mat, 1))
    @printf("  Use threads with HMatrix: ")
    @printf("%s", HMatrices.use_threads() ? "true\n" : "false\n")
    @printf("  Use global index with HMatrix: ")
    @printf("%s", HMatrices.use_global_index() ? "true\n" : "false\n")
    @printf("  Linear solver: ")
    @printf("%s\n", solver.l_solver)
    @printf("  Preconditioner: ")
    @printf("%s", solver.pc ? "ILU\n\n" : "none\n\n")
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