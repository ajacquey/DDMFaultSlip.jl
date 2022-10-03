mutable struct DDSolver{R,T<:Real}
    # " The Problem this solver will act on"
    # problem::AbstractDDProblem{T}

    " The jacobian matrix"
    mat::DDJacobian{R,T}

    " The residual vector"
    rhs::Vector{T}

    " The solution vector"
    solution::Vector{T}

    " Whether or not this Solver has been initialized"
    initialized::Bool

    " Maximum number of nonlinear iterations"
    nl_max_it::Int64

    " Nonlinear absolute tolerance"
    nl_abs_tol::T

    " Nonlinear relative tolerance"
    nl_rel_tol::T

    " Constructor for NormalDDProblem"
    function DDSolver(problem::NormalDDProblem{T}; hmat_eta::T = 3.0, hmat_atol::T = 0.0, nl_abs_tol::T, nl_rel_tol::T, nl_max_it::Int) where {T<:Real}
        mat = NormalDDJacobian(problem; eta = hmat_eta, atol = hmat_atol)
        n_dof = size(mat, 1)
        
        R = typeof(mat.En.coltree)
        return new{R,T}(mat, zeros(T, n_dof), zeros(T, n_dof), false, nl_max_it, nl_abs_tol, nl_rel_tol)
    end

    " Constructor for ShearDDProblem2D"
    function DDSolver(problem::ShearDDProblem2D{T}; hmat_eta::T = 3.0, hmat_atol::T = 0.0, nl_abs_tol::T, nl_rel_tol::T, nl_max_it::Int) where {T<:Real}
        mat = ShearDDJacobian2D(problem; eta = hmat_eta, atol = hmat_atol)
        n_dof = size(mat, 1)

        R = typeof(mat.Es.coltree)
        return new{R,T}(mat, zeros(T, n_dof), zeros(T, n_dof), false, nl_max_it, nl_abs_tol, nl_rel_tol)
    end

    " Constructor for ShearDDProblem3D"
    function DDSolver(problem::ShearDDProblem3D{T}; hmat_eta::T = 3.0, hmat_atol::T = 0.0, nl_abs_tol::T, nl_rel_tol::T, nl_max_it::Int) where {T<:Real}
        mat = ShearDDJacobian3D(problem; eta = hmat_eta, atol = hmat_atol)
        n_dof = size(mat, 1)

        R = typeof(mat.Esxx.coltree)
        return new{R,T}(mat, zeros(T, n_dof), zeros(T, n_dof), false, nl_max_it, nl_abs_tol, nl_rel_tol)
    end

    # " Constructor for CoupledDDProblem"
    # function DDSolver(problem::CoupledDDProblem{T}; hmat_eta::T = 3.0, hmat_atol::T = 0.0, nl_abs_tol::T, nl_rel_tol::T, nl_max_it::Int) where {T<:Real}
    #     if isa(problem.mesh, DDMesh1D)
    #         n_dof = 2 * length(problem.elems)
    #         # mat = 
    #     elseif isa(problem.mesh, DDMesh2D)
    #         n_dof = 3 * length(problem.elems)
    #         # mat = 
    #     end
    #     n_dof = size(mat, 1)

    #     R = typeof(mat.En.coltree)
    #     return new{R,T}(mat, zeros(T, n_dof), zeros(T, n_dof), false, nl_max_it, nl_abs_tol, nl_rel_tol)
    # end
end

function linear_solve!(dx::Vector{T}, solver::DDSolver{R,T}) where {R,T<:Real}
    dx, ch = bicgstabl!(dx, solver.jacobian, -solver.rhs; log = true, verbose = false, abstol = solver.l_abs_tol, reltol = solver.l_rel_tol)

    if log
        if ch.isconverged
            println("    -> Linear Solve converged after ", ch.iters, " iterations.")
        else                
            println("    -> Linear Solve did NOT converge after ", ch.iters, " iterations.")
        end
    end

    return dx
end

" Solve the problem using the IterativeSolvers package"
function solve!(solver::DDSolver{R,T}, problem::AbstractDDProblem{T}, timer::TimerOutput; log::Bool = true) where {R,T<:Real}
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
        println("  ", 0, " Nonlinear Iteration: |R| = ", r0)
    end
    # Main loop
    while (nl_iter <= solver.nl_max_it)
        # Check convergence
        if (r <= solver.nl_abs_tol)
            if log
                println("Nonlinear Solve converged with absolute tolerance!")
                println()
            end
            return nothing
        end
        if (r / r0 <= solver.nl_rel_tol)
            if log
                println("Nonlinear Solve converged with relative tolerance!")
                println()
            end
            return nothing
        end
        # Linear Solve
        @timeit timer "Solve" dx, ch = bicgstabl!(dx, solver.mat, -solver.rhs; log = true, verbose = false, abstol = 1.0e-10, reltol = 1.0e-10)
        # @timeit timer "Solve" dx = jacobi!(dx, solver.mat, -solver.rhs; maxiter = 200)
        if log
            if ch.isconverged
                println("    -> Linear Solve converged after ", ch.iters, " iterations")
            else
                println("    -> Linear Solve did NOT converge after ", ch.iters, " iterations")
            end
        end

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
            println("  ", nl_iter, " Nonlinear Iteration: |R| = ", norm(r))
        end
    end
    # Error if exceeded maximum number of iterations
    if (nl_iter > solver.nl_max_it)
        throw(ErrorException("Exceeded the maximum number of nonlinear iterations!"))
    end
end