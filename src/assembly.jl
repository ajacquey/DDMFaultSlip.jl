function assembleResidualAndJacobian!(solver::DDSolver{R,T}, problem::AbstractDDProblem{T}, timer::TimerOutput) where {R,T<:Real}
    # Reset local jacobian
    @timeit timer "Reinitialize Solver" reinit!(solver; end_time_step=false)

    @timeit timer "Assembly" begin
        # Residuals = collocation stress - imposed stress
        # Collocation stress
        @timeit timer "Collocation" collocation_mul!(solver.rhs, solver.mat, solver.solution)
        # Fluid coupling
        if hasFluidCoupling(problem)
            @timeit timer "Fluid coupling" assembleFluidCouplingResidualAndJacobian!(solver, problem)
        end
        # Imposed constraints
        if hasConstraint(problem)
            @timeit timer "Constraints" assembleConstraintsResidualAndJacobian!(solver, problem)
        end
        # Imposed Frictional constraints
        if hasFrictionConstraint(problem)
            @timeit timer "Frictional constraints" assembleFrictionResidualAndJacobian!(solver, problem)
        end
        # Imposed Cohesive Zone constraints
        if hasCohesiveZoneConstraint(problem)
            @timeit timer "Cohesive zone constraints" assembleCohesiveZoneResidualAndJacobian!(solver, problem)
        end
    end
    return nothing
end

# Constraints
function assembleConstraintsResidualAndJacobian!(solver::DDSolver{R,T}, problem::NormalDDProblem{T}) where {R,T<:Real}
    # Loop over elements
    Threads.@threads for idx in eachindex(problem.mesh.elems)
        for cst in problem.constraints_ϵ
            @inbounds (solver.rhs[idx], solver.mat.jac_loc[idx]) = (solver.rhs[idx], solver.mat.jac_loc[idx]) .- computeConstraints(cst, 0.0, problem.mesh.elems[idx].X)
        end
    end
    return nothing
end

function assembleConstraintsResidualAndJacobian!(solver::DDSolver{R,T}, problem::ShearDDProblem2D{T}) where {R,T<:Real}
    # Loop over elements
    Threads.@threads for idx in eachindex(problem.mesh.elems)
        for cst in problem.constraints_δ
            @inbounds (solver.rhs[idx], solver.mat.jac_loc[idx]) = (solver.rhs[idx], solver.mat.jac_loc[idx]) .- computeConstraints(cst, 0.0, problem.mesh.elems[idx].X)
        end
    end
    return nothing
end

function assembleConstraintsResidualAndJacobian!(solver::DDSolver{R,T}, problem::ShearDDProblem3D{T}) where {R,T<:Real}
    # Loop over elements
    Threads.@threads for idx in eachindex(problem.mesh.elems)
        for cst in problem.constraints_δx
            @inbounds (solver.rhs[idx], solver.mat.jac_loc_x[1][idx]) = (solver.rhs[idx], solver.mat.jac_loc_x[1][idx]) .- computeConstraints(cst, 0.0, problem.mesh.elems[idx].X)
        end

        for cst in problem.constraints_δy
            @inbounds (solver.rhs[problem.n+idx], solver.mat.jac_loc_y[2][idx]) = (solver.rhs[problem.n+idx], solver.mat.jac_loc_y[2][idx]) .- computeConstraints(cst, 0.0, problem.mesh.elems[idx].X)
        end
    end
    return nothing
end

function assembleConstraintsResidualAndJacobian!(solver::DDSolver{R,T}, problem::CoupledDDProblem2D{T}) where {R,T<:Real}
    # Loop over elements
    Threads.@threads for idx in eachindex(problem.mesh.elems)
        for cst in problem.constraints_ϵ
            @inbounds (solver.rhs[idx], solver.mat.jac_loc_ϵ[1][idx]) = (solver.rhs[idx], solver.mat.jac_loc_ϵ[1][idx]) .- computeConstraints(cst, 0.0, problem.mesh.elems[idx].X)
        end

        for cst in problem.constraints_δ
            @inbounds (solver.rhs[problem.n+idx], solver.mat.jac_loc_δ[2][idx]) = (solver.rhs[problem.n+idx], solver.mat.jac_loc_δ[2][idx]) .- computeConstraints(cst, 0.0, problem.mesh.elems[idx].X)
        end
    end

    return nothing
end

function assembleConstraintsResidualAndJacobian!(solver::DDSolver{R,T}, problem::CoupledDDProblem3D{T}) where {R,T<:Real}
    # Loop over elements
    Threads.@threads for idx in eachindex(problem.mesh.elems)
        for cst in problem.constraints_ϵ
            @inbounds (solver.rhs[idx], solver.mat.jac_loc_ϵ[1][idx]) = (solver.rhs[idx], solver.mat.jac_loc_ϵ[1][idx]) .- computeConstraints(cst, 0.0, problem.mesh.elems[idx].X)
        end

        for cst in problem.constraints_δx
            @inbounds (solver.rhs[problem.n+idx], solver.mat.jac_loc_δx[2][idx]) = (solver.rhs[problem.n+idx], solver.mat.jac_loc_δx[2][idx]) .- computeConstraints(cst, 0.0, problem.mesh.elems[idx].X)
        end

        for cst in problem.constraints_δy
            @inbounds (solver.rhs[2*problem.n+idx], solver.mat.jac_loc_δy[3][idx]) = (solver.rhs[2*problem.n+idx], solver.mat.jac_loc_δx[3][idx]) .- computeConstraints(cst, 0.0, problem.mesh.elems[idx].X)
        end
    end

    return nothing
end

function assembleFluidCouplingResidualAndJacobian!(solver::DDSolver{R,T}, problem::AbstractDDProblem{T}) where {R,T<:Real}
    Threads.@threads for idx in eachindex(problem.mesh.elems)
        @inbounds solver.rhs[idx] -= problem.fluid_coupling[1].p[idx] - problem.fluid_coupling[1].p_old[idx]
    end

    return nothing
end

# Frictional constraints
function assembleFrictionResidualAndJacobian!(solver::DDSolver{R,T}, problem::AbstractDDProblem{T}) where {R,T<:Real}
    return nothing
end

function assembleFrictionResidualAndJacobian!(solver::DDSolver{R,T}, problem::CoupledDDProblem2D{T}) where {R,T<:Real}
    Threads.@threads for idx in eachindex(problem.mesh.elems)
        @inbounds (Res, Jac) = applyFrictionalConstraints(problem.friction[1], SVector(problem.ϵ.value[idx] - problem.ϵ.value_old[idx], problem.δ.value[idx] - problem.δ.value_old[idx]), SVector(problem.ϵ.value_old[idx], problem.δ.value_old[idx]), SVector(problem.σ.value_old[idx], problem.τ.value_old[idx]))

        @inbounds solver.rhs[idx] -= Res[1]
        @inbounds solver.rhs[problem.n+idx] -= Res[2]
        @inbounds solver.mat.jac_loc_ϵ[1][idx] -= Jac[1, 1]
        @inbounds solver.mat.jac_loc_ϵ[2][idx] -= Jac[2, 1]
        @inbounds solver.mat.jac_loc_δ[1][idx] -= Jac[1, 2]
        @inbounds solver.mat.jac_loc_δ[2][idx] -= Jac[2, 2]
    end

    return nothing
end

function assembleFrictionResidualAndJacobian!(solver::DDSolver{R,T}, problem::CoupledDDProblem3D{T}) where {R,T<:Real}
    Threads.@threads for idx in eachindex(problem.mesh.elems)
        @inbounds (Res, Jac) = applyFrictionalConstraints(problem.friction[1], SVector(problem.ϵ.value[idx] - problem.ϵ.value_old[idx], problem.δ_x.value[idx] - problem.δ_x.value_old[idx], problem.δ_y.value[idx] - problem.δ_y.value_old[idx]), SVector(problem.ϵ.value_old[idx], problem.δ_x.value_old[idx], problem.δ_y.value_old[idx]), SVector(problem.σ.value_old[idx], problem.τ_x.value_old[idx], problem.τ_y.value_old[idx]))

        @inbounds solver.rhs[idx] -= Res[1]
        @inbounds solver.rhs[problem.n+idx] -= Res[2]
        @inbounds solver.rhs[2*problem.n+idx] -= Res[3]
        @inbounds solver.mat.jac_loc_ϵ[1][idx] -= Jac[1, 1]
        @inbounds solver.mat.jac_loc_ϵ[2][idx] -= Jac[2, 1]
        @inbounds solver.mat.jac_loc_ϵ[3][idx] -= Jac[3, 1]
        @inbounds solver.mat.jac_loc_δx[1][idx] -= Jac[1, 2]
        @inbounds solver.mat.jac_loc_δx[2][idx] -= Jac[2, 2]
        @inbounds solver.mat.jac_loc_δx[3][idx] -= Jac[3, 2]
        @inbounds solver.mat.jac_loc_δy[1][idx] -= Jac[1, 3]
        @inbounds solver.mat.jac_loc_δy[2][idx] -= Jac[2, 3]
        @inbounds solver.mat.jac_loc_δy[3][idx] -= Jac[3, 3]
    end

    return nothing
end

# Cohesive zone constraints
function assembleCohesiveZoneResidualAndJacobian!(solver::DDSolver{R,T}, problem::AbstractDDProblem{T}) where {R,T<:Real}
    return nothing
end

function assembleCohesiveZoneResidualAndJacobian!(solver::DDSolver{R,T}, problem::NormalDDProblem{T}) where {R,T<:Real}
    Threads.@threads for idx in eachindex(problem.mesh.elems)
        @inbounds (Res, Jac) = applyCohesiveZoneConstraints(problem.cohesive[1], problem.ϵ.value[idx] - problem.ϵ.value_old[idx], problem.mesh.elems[idx].X, problem.ϵ.value_old[idx], problem.σ.value_old[idx])

        @inbounds solver.rhs[idx] -= Res
        @inbounds solver.mat.jac_loc[idx] -= Jac
    end

    return nothing
end

function assembleCohesiveZoneResidualAndJacobian!(solver::DDSolver{R,T}, problem::ShearDDProblem3D{T}) where {R,T<:Real}
    Threads.@threads for idx in eachindex(problem.mesh.elems)
        @inbounds (Res, Jac) = applyCohesiveZoneConstraints(problem.cohesive[1], SVector(problem.δ_x.value[idx] - problem.δ_x.value_old[idx], problem.δ_y.value[idx] - problem.δ_y.value_old[idx]), problem.mesh.elems[idx].X, SVector(problem.δ_x.value_old[idx], problem.δ_y.value_old[idx]), SVector(problem.τ_x.value_old[idx], problem.τ_y.value_old[idx]))

        @inbounds solver.rhs[idx] -= Res[1]
        @inbounds solver.rhs[problem.n+idx] -= Res[2]
        @inbounds solver.mat.jac_loc_x[1][idx] -= Jac[1, 1]
        @inbounds solver.mat.jac_loc_x[2][idx] -= Jac[2, 1]
        @inbounds solver.mat.jac_loc_y[1][idx] -= Jac[1, 2]
        @inbounds solver.mat.jac_loc_y[2][idx] -= Jac[2, 2]
    end

    return nothing
end

function update!(problem::NormalDDProblem{T}, solver::DDSolver{R,T}) where {R,T<:Real}
    # Loop over elements
    Threads.@threads for idx in eachindex(problem.mesh.elems)
        @inbounds problem.ϵ.value[idx] = problem.ϵ.value_old[idx] + solver.solution[idx]
    end
    problem.σ.value = problem.σ.value_old + collocation_mul!(zeros(T, size(solver.solution)), solver.mat, solver.solution)
    # Fluid coupling
    if hasFluidCoupling(problem)
        problem.σ.value -= problem.fluid_coupling[1].p + problem.fluid_coupling[1].p_old
    end

    return nothing
end

function update!(problem::ShearDDProblem2D{T}, solver::DDSolver{R,T}) where {R,T<:Real}
    # Loop over elements
    Threads.@threads for idx in eachindex(problem.mesh.elems)
        @inbounds problem.δ.value[idx] = problem.δ.value_old[idx] + solver.solution[idx]
    end
    problem.τ.value = problem.τ.value_old + collocation_mul!(zeros(T, size(solver.solution)), solver.mat, solver.solution)
    return nothing
end

function update!(problem::ShearDDProblem3D{T}, solver::DDSolver{R,T}) where {R,T<:Real}
    # Loop over elements
    Threads.@threads for idx in eachindex(problem.mesh.elems)
        @inbounds problem.δ_x.value[idx] = problem.δ_x.value_old[idx] + solver.solution[idx]
        @inbounds problem.δ_y.value[idx] = problem.δ_y.value_old[idx] + solver.solution[problem.n+idx]
    end
    problem.τ_x.value = problem.τ_x.value_old + collocation_mul(solver.mat, solver.solution, 1)
    problem.τ_y.value = problem.τ_y.value_old + collocation_mul(solver.mat, solver.solution, 2)
    return nothing
end

function update!(problem::CoupledDDProblem2D{T}, solver::DDSolver{R,T}) where {R,T<:Real}
    # Loop over elements
    Threads.@threads for idx in eachindex(problem.mesh.elems)
        @inbounds problem.ϵ.value[idx] = problem.ϵ.value_old[idx] + solver.solution[idx]
        @inbounds problem.δ.value[idx] = problem.δ.value_old[idx] + solver.solution[problem.n+idx]
    end
    problem.σ.value = problem.σ.value_old + collocation_mul(solver.mat, solver.solution, 0)
    problem.τ.value = problem.τ.value_old + collocation_mul(solver.mat, solver.solution, 1)
    # Fluid coupling
    if hasFluidCoupling(problem)
        problem.σ.value -= (problem.fluid_coupling[1].p - problem.fluid_coupling[1].p_old)
    end

    return nothing
end

function update!(problem::CoupledDDProblem3D{T}, solver::DDSolver{R,T}) where {R,T<:Real}
    # Loop over elements
    Threads.@threads for idx in eachindex(problem.mesh.elems)
        @inbounds problem.ϵ.value[idx] = problem.ϵ.value_old[idx] + solver.solution[idx]
        @inbounds problem.δ_x.value[idx] = problem.δ_x.value_old[idx] + solver.solution[problem.n+idx]
        @inbounds problem.δ_y.value[idx] = problem.δ_y.value_old[idx] + solver.solution[2*problem.n+idx]
    end
    problem.σ.value = problem.σ.value_old + collocation_mul(solver.mat, solver.solution, 0)
    problem.τ_x.value = problem.τ_x.value_old + collocation_mul(solver.mat, solver.solution, 1)
    problem.τ_y.value = problem.τ_y.value_old + collocation_mul(solver.mat, solver.solution, 2)
    # Fluid coupling
    if hasFluidCoupling(problem)
        problem.σ.value -= (problem.fluid_coupling[1].p - problem.fluid_coupling[1].p_old)
    end

    return nothing
end

function computePressureCoupling!(problem::AbstractDDProblem{T}, time::T, timer::TimerOutput) where {T<:Real}
    # Check if problem has pressure coupling
    if hasFluidCoupling(problem)
        @timeit timer "Pressure update" begin
            updatePressure!(problem.fluid_coupling[1], [problem.mesh.elems[idx].X for idx in eachindex(problem.mesh.elems)], time)
        end
    end

    return nothing
end