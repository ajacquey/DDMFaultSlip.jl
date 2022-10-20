function assembleResidualAndJacobian!(solver::DDSolver{R,T}, problem::AbstractDDProblem{T}, timer::TimerOutput) where {R,T<:Real}
    # Reset local jacobian
    @timeit timer "Reinitialize Solver" reinit!(solver; end_time_step = false)

    @timeit timer "Assembly" begin
        # Residuals = collocation stress - imposed stress
        # Collocation stress
        @timeit timer "Collocation" collocation_mul!(solver.rhs, solver.mat, solver.solution)
        # Imposed constraints
        @timeit timer "Constraints" assembleConstraintsResidualAndJacobian!(solver, problem)
        # Fluid coupling
        @timeit timer "Fluid coupling" assembleFluidCouplingResidualAndJacobian!(solver, problem)
        # Imposed Friction
        @timeit timer "Frictional constraints" assembleFrictionResidualAndJacobian!(solver, problem)
    end
    return nothing
end

function assembleConstraintsResidualAndJacobian!(solver::DDSolver{R,T}, problem::NormalDDProblem{T}) where {R,T<:Real}
    # Loop over elements
    Threads.@threads for idx in 1:length(problem.mesh.elems)
        for cst in problem.constraints
            (solver.rhs[idx], solver.mat.jac_loc[idx]) = (solver.rhs[idx], solver.mat.jac_loc[idx]) .- computeConstraints(cst, 0.0, problem.mesh.elems[idx].X)
        end
    end
    return nothing
end

function assembleConstraintsResidualAndJacobian!(solver::DDSolver{R,T}, problem::ShearDDProblem2D{T}) where {R,T<:Real}
    # Loop over elements
    Threads.@threads for idx in 1:length(problem.mesh.elems)
        for cst in problem.constraints
            (solver.rhs[idx], solver.mat.jac_loc[idx]) = (solver.rhs[idx], solver.mat.jac_loc[idx]) .- computeConstraints(cst, 0.0, problem.mesh.elems[idx].X)
        end
    end
    return nothing
end

function assembleConstraintsResidualAndJacobian!(solver::DDSolver{R,T}, problem::ShearDDProblem3D{T}) where {R,T<:Real}
    # Loop over elements
    n = size(solver.mat.Esxx, 1)
    Threads.@threads for idx in 1:length(problem.mesh.elems)
        for cst in problem.constraints_x
            (solver.rhs[idx], solver.mat.jac_loc_x[idx]) = (solver.rhs[idx], solver.mat.jac_loc_x[idx]) .- computeConstraints(cst, 0.0, problem.mesh.elems[idx].X)
        end

        for cst in problem.constraints_y
            (solver.rhs[n + idx], solver.mat.jac_loc_y[idx]) = (solver.rhs[n + idx], solver.mat.jac_loc_y[idx]) .- computeConstraints(cst, 0.0, problem.mesh.elems[idx].X)
        end
    end
    return nothing
end

function assembleConstraintsResidualAndJacobian!(solver::DDSolver{R,T}, problem::CoupledDDProblem2D{T}) where {R,T<:Real}
    # Loop over elements
    n = size(solver.mat.E, 1)
    Threads.@threads for idx in 1:length(problem.mesh.elems)
        for cst in problem.constraints_ϵ
            (solver.rhs[idx], solver.mat.jac_loc_ϵ[1][idx]) = (solver.rhs[idx], solver.mat.jac_loc_ϵ[1][idx]) .- computeConstraints(cst, 0.0, problem.mesh.elems[idx].X)
        end

        for cst in problem.constraints_δ
            (solver.rhs[n + idx], solver.mat.jac_loc_δ[2][idx]) = (solver.rhs[n + idx], solver.mat.jac_loc_δ[2][idx]) .- computeConstraints(cst, 0.0, problem.mesh.elems[idx].X)
        end
    end

    return nothing
end

function assembleFluidCouplingResidualAndJacobian!(solver::DDSolver{R,T}, problem::AbstractDDProblem{T}) where {R,T<:Real}
   # Check if problem has fluid coupling
   if hasFluidCoupling(problem)
        Threads.@threads for idx in 1:length(problem.mesh.elems)
            solver.rhs[idx] -= problem.fluid_coupling[1].p[idx] - problem.fluid_coupling[1].p_old[idx]
        end
    end

    return nothing
end

function assembleFrictionResidualAndJacobian!(solver::DDSolver{R,T}, problem::AbstractDDProblem{T}) where {R,T<:Real}
    return nothing
end

function assembleFrictionResidualAndJacobian!(solver::DDSolver{R,T}, problem::CoupledDDProblem2D{T}) where {R,T<:Real}
    # Check if problem has frictional constraints
    if hasFrictionConstraint(problem)
        n = size(solver.mat.E, 1)
        Threads.@threads for idx in 1:length(problem.mesh.elems)
            ((rϵ, rδ), (jϵϵ, jϵδ, jδϵ, jδδ)) = applyFrictionalConstraints(problem.friction[1], SVector(problem.ϵ.value[idx], problem.δ.value[idx]), SVector(problem.σ.value_old[idx], problem.τ.value_old[idx]))

            solver.rhs[idx] -= rϵ
            solver.rhs[n + idx] -= rδ
            solver.mat.jac_loc_ϵ[1][idx] -= jϵϵ
            solver.mat.jac_loc_ϵ[2][idx] -= jϵδ
            solver.mat.jac_loc_δ[1][idx] -= jδϵ
            solver.mat.jac_loc_δ[2][idx] -= jδδ
        end
    end

    return nothing
end

function update!(problem::NormalDDProblem{T}, solver::DDSolver{R,T}) where {R,T<:Real}
    # Loop over elements
    Threads.@threads for idx in 1:length(problem.mesh.elems)
        problem.ϵ.value[idx] = problem.ϵ.value_old[idx] + solver.solution[idx]
    end
    problem.σ.value = problem.σ.value_old + collocation_mul!(similar(solver.solution), solver.mat, solver.solution)
    # Fluid coupling
    if hasFluidCoupling(problem)
        problem.σ.value -= problem.fluid_coupling[1].p + problem.fluid_coupling[1].p_old
    end

    return nothing
end

function update!(problem::ShearDDProblem2D{T}, solver::DDSolver{R,T}) where {R,T<:Real}
    # Loop over elements
    Threads.@threads for idx in 1:length(problem.mesh.elems)
        problem.δ.value[idx] = problem.δ.value_old[idx] + solver.solution[idx]
    end
    problem.τ.value = problem.τ.value_old + collocation_mul!(similar(solver.solution), solver.mat, solver.solution)
    return nothing
end

function update!(problem::ShearDDProblem3D{T}, solver::DDSolver{R,T}) where {R,T<:Real}
    # Loop over elements
    n = size(solver.mat.Esxx, 1)
    Threads.@threads for idx in 1:length(problem.mesh.elems)
        problem.δ_x.value[idx] = problem.δ_x.value_old[idx] + solver.solution[idx]
        problem.δ_y.value[idx] = problem.δ_y.value_old[idx] + solver.solution[n+idx]
    end
    problem.τ_x.value = problem.τ_x.value_old + collocation_mul(solver.mat, solver.solution, 1)
    problem.τ_y.value = problem.τ_y.value_old + collocation_mul(solver.mat, solver.solution, 2)
    return nothing
end

function update!(problem::CoupledDDProblem2D{T}, solver::DDSolver{R,T}) where {R,T<:Real}
    # Loop over elements
    n = size(solver.mat.E, 1)
    Threads.@threads for idx in 1:length(problem.mesh.elems)
        problem.ϵ.value[idx] = problem.ϵ.value_old[idx] + solver.solution[idx]
        problem.δ.value[idx] = problem.δ.value_old[idx] + solver.solution[n+idx]
    end
    problem.σ.value = problem.σ.value_old + collocation_mul(solver.mat, solver.solution, 0)
    problem.τ.value = problem.τ.value_old + collocation_mul(solver.mat, solver.solution, 1)
    # Fluid coupling
    if hasFluidCoupling(problem)
        problem.σ.value -= problem.fluid_coupling[1].p + problem.fluid_coupling[1].p_old
    end

    return nothing
end

function computePressureCoupling!(problem::AbstractDDProblem{T}, time::T, timer::TimerOutput) where {T<:Real}
    # Check if problem has pressure coupling
    if hasFluidCoupling(problem)
        @timeit timer "Pressure update" begin
            # Loop over elements
            Threads.@threads for idx in 1:length(problem.mesh.elems)
                updatePressure!(problem.fluid_coupling[1], problem.mesh.elems[idx].X, time, idx)
            end
        end
    end
    
    return nothing
end