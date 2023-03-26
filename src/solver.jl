mutable struct DDSolver{R,T<:Real}
    " The jacobian matrix"
    mat::AbstractDDJacobian{R,T}

    " The residual vector"
    rhs::Vector{T}

    " The solution vector"
    solution::Vector{T}

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
        hmat_eta::T=3.0, hmat_atol::T=1.0e-06,
        pc::Bool=true, pc_atol::T=1.0e-2,
        nl_abs_tol::T, nl_rel_tol::T, nl_max_it::Int,
        l_solver::String, l_max_it::Int, l_abs_tol::T, l_rel_tol::T) where {T<:Real}
        mat = NormalDDJacobian(problem; eta=hmat_eta, atol=hmat_atol, pc=pc, pc_atol=pc_atol)
        n_dof = size(mat, 1)

        # Check if linear solver is provided
        checkLinearSolver(l_solver)

        R = typeof(mat.En.coltree)
        return new{R,T}(mat, zeros(T, n_dof), zeros(T, n_dof),
            nl_max_it, nl_abs_tol, nl_rel_tol,
            l_solver, l_max_it, l_abs_tol, l_rel_tol)
    end

    " Constructor for ShearDDProblem2D"
    function DDSolver(problem::ShearDDProblem2D{T};
        hmat_eta::T=3.0, hmat_atol::T=1.0e-06,
        pc::Bool=true, pc_atol::T=1.0e-2,
        nl_abs_tol::T, nl_rel_tol::T, nl_max_it::Int,
        l_solver::String, l_max_it::Int, l_abs_tol::T, l_rel_tol::T) where {T<:Real}
        mat = ShearDDJacobian2D(problem; eta=hmat_eta, atol=hmat_atol, pc=pc, pc_atol=pc_atol)
        n_dof = size(mat, 1)

        # Check if linear solver is provided
        checkLinearSolver(l_solver)

        R = typeof(mat.Es.coltree)
        return new{R,T}(mat, zeros(T, n_dof), zeros(T, n_dof),
            nl_max_it, nl_abs_tol, nl_rel_tol,
            l_solver, l_max_it, l_abs_tol, l_rel_tol)
    end

    " Constructor for ShearDDProblem3D"
    function DDSolver(problem::ShearDDProblem3D{T};
        hmat_eta::T=3.0, hmat_atol::T=1.0e-06,
        pc::Bool=true, pc_atol::T=1.0e-2,
        nl_abs_tol::T, nl_rel_tol::T, nl_max_it::Int,
        l_solver::String, l_max_it::Int, l_abs_tol::T, l_rel_tol::T) where {T<:Real}
        if (problem.ν != 0.0)
            mat = ShearDDJacobian3D(problem; eta=hmat_eta, atol=hmat_atol, pc=pc, pc_atol=pc_atol)
        else
            mat = ShearNoNuDDJacobian3D(problem; eta=hmat_eta, atol=hmat_atol, pc=pc, pc_atol=pc_atol)
        end
        n_dof = size(mat, 1)

        # Check if linear solver is provided
        checkLinearSolver(l_solver)

        R = typeof(mat.Es.coltree)
        return new{R,T}(mat, zeros(T, n_dof), zeros(T, n_dof),
            nl_max_it, nl_abs_tol, nl_rel_tol,
            l_solver, l_max_it, l_abs_tol, l_rel_tol)
    end

    " Constructor for CoupledDDProblem2D"
    function DDSolver(problem::CoupledDDProblem2D{T}; 
        hmat_eta::T=3.0, hmat_atol::T=1.0e-06,
        pc::Bool=true, pc_atol::T=1.0e-2,
        nl_abs_tol::T, nl_rel_tol::T, nl_max_it::Int,
        l_solver::String, l_max_it::Int, l_abs_tol::T, l_rel_tol::T) where {T<:Real}
        mat = CoupledDDJacobian2D(problem; eta=hmat_eta, atol=hmat_atol, pc=pc, pc_atol=pc_atol)
        n_dof = size(mat, 1)

        # Check if linear solver is provided
        checkLinearSolver(l_solver)

        R = typeof(mat.E.coltree)
        return new{R,T}(mat, zeros(T, n_dof), zeros(T, n_dof),
            nl_max_it, nl_abs_tol, nl_rel_tol,
            l_solver, l_max_it, l_abs_tol, l_rel_tol)
    end

    " Constructor for CoupledDDProblem3D"
    function DDSolver(problem::CoupledDDProblem3D{T}; 
        hmat_eta::T=3.0, hmat_atol::T=1.0e-06,
        pc::Bool=true, pc_atol::T=1.0e-2,
        nl_abs_tol::T, nl_rel_tol::T, nl_max_it::Int,
        l_solver::String, l_max_it::Int, l_abs_tol::T, l_rel_tol::T) where {T<:Real}
        if (problem.ν != 0.0)
            mat = CoupledDDJacobian3D(problem; eta=hmat_eta, atol=hmat_atol)
        else
            mat = CoupledNoNuDDJacobian3D(problem; eta=hmat_eta, atol=hmat_atol)
        end
        n_dof = size(mat, 1)

        # Check if linear solver is provided
        checkLinearSolver(l_solver)

        R = typeof(mat.E.coltree)
        return new{R,T}(mat, zeros(T, n_dof), zeros(T, n_dof),
            nl_max_it, nl_abs_tol, nl_rel_tol,
            l_solver, l_max_it, l_abs_tol, l_rel_tol)
    end
end

function linear_solve!(dx::Vector{T}, solver::DDSolver{R,T}, log::Bool) where {R,T<:Real}
    if solver.l_solver == "bicgstabl"
        dx, ch = bicgstabl!(dx, solver.mat, -solver.rhs; Pl=solver.mat.Pl, log=true, abstol=solver.l_abs_tol, reltol=solver.l_rel_tol, max_mv_products=4*solver.l_max_it)
    elseif solver.l_solver == "gmres"
        dx, ch = gmres!(dx, solver.mat, -solver.rhs; Pl=solver.mat.Pl, log=true, abstol=solver.l_abs_tol, reltol=solver.l_rel_tol, maxiter=solver.l_max_it)
    elseif solver.l_solver == "idrs"
        dx, ch = idrs!(dx, solver.mat, -solver.rhs; Pl=solver.mat.Pl, log=true, abstol=solver.l_abs_tol, reltol=solver.l_rel_tol, maxiter=solver.l_max_it)
    end

    if log
        if ch.isconverged
            @printf("    -> Linear Solve converged after %i iterations.\n", ch.iters)
        else
            @printf("    -> Linear Solve did NOT converge after %i iterations.\n", ch.iters)
        end
    end

    return dx
end

function print_NL_res(it::Int, r::T) where {T<:Real}
    @printf("  %i Nonlinear |R| = %e\n", it, r)
    return nothing
end

" Solve the problem using the IterativeSolvers package"
function solve!(solver::DDSolver{R,T}, problem::AbstractDDProblem{T}, timer::TimerOutput; log::Bool=true, linear_log::Bool=false)::Bool where {R,T<:Real}
    ##### Newton loop #####
    # Non-linear iterations
    nl_iter = 0
    # Declare solution
    dx = zeros(T, length(solver.rhs))
    # Initial residual
    assembleResidualAndJacobian!(solver, problem, timer)
    r = norm(solver.rhs)
    r0 = r
    # Preconditioner 
    # @timeit timer "Preconditionning" precond = ilu(solver.mat, τ = 0.01)
    # @timeit timer "Preconditionning" precond = JacobiPreconditioner(solver.mat)
    if log
        print_NL_res(0, r0)
    end
    # Main loop
    while (nl_iter <= solver.nl_max_it)
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
        # Linear Solve
        @timeit timer "Solve" dx = linear_solve!(dx, solver, linear_log)

        # Update solution
        solver.solution .+= dx
        # Update problem
        @timeit timer "Update problem" update!(problem, solver)
        # Update residuals and jacobian
        assembleResidualAndJacobian!(solver, problem, timer)
        r = norm(solver.rhs)
        # Preconditioner 
        # @timeit timer "Preconditionning" precond = ilu(solver.mat, τ = 0.01)

        nl_iter += 1
        if log
            print_NL_res(nl_iter, norm(r))
        end
    end
    # Error if exceeded maximum number of iterations
    if (nl_iter > solver.nl_max_it)
        @printf("Solve diverged: exceeded the maximum number of nonlinear iterations!\n")
        return false
    end
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
    @printf("%s", isa(solver.mat.Pl, Identity) ? "none\n\n" : "ILU\n\n")
end

" Reinitialize solver after successful solve"
function reinit!(solver::DDSolver{R,T}; end_time_step::Bool=false) where {R,T<:Real}
    fill!(solver.rhs, 0.0)
    reinitLocalJacobian!(solver.mat)
    if end_time_step
        fill!(solver.solution, 0.0)
    end
end