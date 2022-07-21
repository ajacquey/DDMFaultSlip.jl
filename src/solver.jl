abstract type DDMSolver{T<:Real} end

mutable struct DirectDDMSolver{R,T<:Real} <: DDMSolver{T}
    jacobian::DDMJacobian{R,T}
    rhs::Vector{T}
    x::Vector{T}
    preconditioner::Bool
    nl_abs_tol::T
    nl_rel_tol::T
    nl_max_it::T
    # Constructor for 1D
    function DirectDDMSolver(mesh::Mesh1D{T}, n_var::Int; order::Int = 0, preconditioner::Bool = true, hmat_eta::T=3.0, hmat_atol::T=0.0, nl_abs_tol::T = 1.0e-10, nl_rel_tol::T = 1.0e-10, nl_max_it::Int = 100) where {T<:Real}
        # Assemble Jacobian
        jacobian = assemble_1D_DDM_jacobian(mesh, n_var; order=order, eta=hmat_eta, atol=hmat_atol)

        R = typeof(jacobian.Hmat.coltree)
        return new{R,T}(jacobian, zeros(T, size(jacobian, 1)), zeros(T, size(jacobian, 1)), preconditioner, nl_abs_tol, nl_rel_tol, nl_max_it)
    end
end

function linear_solve!(dx::Vector{T}, solver::DirectDDMSolver{R,T}) where {R,T<:Real}
    # Not sure how to deal with \ here with the type DDMJacobian...
    ErrorException("Direct solver is not supported yet! Please use an iterative solver instead.")

    return dx
end

mutable struct IterativeDDMSolver{R,T<:Real} <: DDMSolver{T}
    jacobian::DDMJacobian{R,T}
    rhs::Vector{T}
    x::Vector{T}
    preconditioner::Bool
    nl_abs_tol::T
    nl_rel_tol::T
    nl_max_it::T
    l_abs_tol::T
    l_rel_tol::T
    l_max_it::T
    # Constructor for 1D
    function IterativeDDMSolver(mesh::Mesh1D{T}, n_var::Int; order::Int = 0, preconditioner::Bool = true, hmat_eta::T = 3.0, hmat_atol::T = 0.0, nl_abs_tol::T = 1.0e-10, nl_rel_tol::T = 1.0e-10, nl_max_it::Int = 100, l_abs_tol::T = 1.0e-10, l_rel_tol::T = 1.0e-10, l_max_it::Int = 100) where {T<:Real}
        # Assemble Jacobian
        jacobian = assemble_1D_DDM_jacobian(mesh, n_var; order=order, eta=hmat_eta, atol=hmat_atol)

        R = typeof(jacobian.Hmat.coltree)
        return new{R,T}(jacobian, zeros(T, size(jacobian, 1)), zeros(T, size(jacobian, 1)), preconditioner, nl_abs_tol, nl_rel_tol, nl_max_it, l_abs_tol, l_rel_tol, l_max_it)
    end
end

function linear_solve!(dx::Vector{T}, solver::IterativeDDMSolver{R,T}) where {R,T<:Real}
    dx, ch = bicgstabl!(dx, solver.jacobian, -solver.rhs; log = true, verbose = false, abstol = solver.l_abs_tol, reltol = solver.l_rel_tol)

    if log
        if ch.isconverged
            println("    -> Linear Solve converged after ", ch.iters, " iterations")
        else                
            println("    -> Linear Solve did NOT converge after ", ch.iters, " iterations")
        end
    end

    return dx
end

function assemble_1D_DDM_jacobian(mesh::Mesh1D{T}, n_var; order::Int = 0, eta::T=3.0, atol::T=0.0) where {T<:Real}
    # Collocation structures
    cps_i = cps_j = Collocation1D(mesh; order=order)

    # Kernel matrix
    K = PWC1DElasticMatrix(cps_i, cps_j)

    return DDMJacobian(K, n_var; eta=eta, atol=atol)
end

function solve!(solver::DDMSolver{T}, timer::TimerOutput; log::Bool = true) where {T<:Real}
    ##### Newton loop #####
    # Non-linear iterations
    nl_iter = 0

    # Declare solution
    dx = zeros(T, length(solver.rhs))

    # Initial residual
    # assembleResidualAndJacobian!(solver, solver.problem, timer)
    r = norm(solver.rhs)
    r0 = r

    if log
        println("  ", 0, " Nonlinear Iteration: |R| = ", r0)
    end

    # Main loop
    while (nl_iter <= solver.nl_max_iters)
        # Check convergence
        if (r <= solver.nl_abs_tol)
            if log
                println("Nonlinear Solve converged with absolute tolerance!")
                println()
            end
            return
        end
        if (r / r0 <= solver.nl_rel_tol)
            if log
                println("Nonlinear Solve converged with relative tolerance!")
                println()
            end
            return
        end

        # Preconditioner 
        # @timeit timer "Preconditionning" precond = ilu(solver.mat, τ = 0.01)

        # Linear Solve
        @timeit timer "Solve" dx = linear_solve!(dx, solver)
        # @timeit timer "Solve" dx, ch = bicgstabl!(dx, solver.mat, -solver.rhs; log = true, verbose = false, abstol = 1.0e-10, reltol = 1.0e-10)
        # @timeit timer "Solve" dx = jacobi!(dx, solver.mat, -solver.rhs; maxiter = 200)

        # Update solution
        solver.solution .+= dx

        # Update problem
        # @timeit timer "Update problem" update!(solver.problem, solver)

        # Update residuals and jacobian
        # assembleResidualAndJacobian!(solver, solver.problem, timer)
        r = norm(solver.rhs)

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