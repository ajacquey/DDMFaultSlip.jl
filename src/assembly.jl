function assembleResidualAndJacobian!(solver::DDSolver{R,T}, problem::AbstractDDProblem{T}, timer::TimerOutput) where {R,T<:Real}
    @timeit timer "Assembly" begin
        # Residuals = collocation stress - imposed stress
        # Collocation stress
        @timeit timer "Collocation" collocation_mul!(solver.rhs, solver.mat, solver.solution)
        # # Imposed constraints
        @timeit timer "Constraints" assembleConstraintsResidualAndJacobian!(solver, problem)
    end
    return nothing
end

function collocationResiduals!(solver::DDSolver{R,T}, problem::ShearDDProblem3D{T}) where {R,T<:Real}
    n = size(problem.mesh.elems)
    solver.rhs[1:n] = solver.mat.Esxx * solver.solution[1:n] + solver.mat.Esxy * solver.solution[n+1:2*n]
    solver.rhs[n+1:2*n] = solver.mat.Esxy * solver.solution[1:n] + solver.mat.Esyy * solver.solution[n+1:2*n]
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

        # for frct in problem.friction
        # end
    end
end

function update!(problem::NormalDDProblem{T}, solver::DDSolver{R,T}) where {R,T<:Real}
    # Loop over elements
    Threads.@threads for idx in 1:length(problem.mesh.elems)
        problem.ϵ.value[idx] = problem.ϵ.value_old[idx] + solver.solution[idx]
    end
    problem.σ.value = problem.σ.value_old + collocation_mul!(similar(solver.solution), solver.mat, solver.solution)
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
    return nothing
end