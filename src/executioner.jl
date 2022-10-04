function run!(problem::AbstractDDProblem{T}; log::Bool = true, nl_max_it::Int64 = 100, nl_abs_tol::T = 1.0e-10, nl_rel_tol::T = 1.0e-10) where {T<:Real}
    # Timer
    timer = TimerOutput()
    
    # Initialize solver
    @timeit timer "Initialize Solver" solver = DDSolver(problem; nl_max_it = nl_max_it, nl_abs_tol = nl_abs_tol, nl_rel_tol = nl_rel_tol)

    # Display some information about simulation
    if log
        @timeit timer "Priting simulation information" begin
            show(problem)
        end
    end

    # Apply ICs
    @timeit timer "Apply Initial Conditions" applyIC!(problem)

    # Steady state problem
    solve!(solver, problem, timer; log)

    # Final update
    # @timeit timer "Final update" final_update!(problem)

    # End of simulation information - TimerOutputs
    if log
        print_timer(timer, title = "Performance graph")
        println()
    end

    return nothing
end