function assembleResidualAndJacobian!(solver::DDSolver{R,T}, problem::AbstractDDProblem{T}, timer::TimerOutput) where {R,T<:Real}
    # Reset local jacobian
    @timeit timer "Reinitialize Solver" reinit!(solver; end_time_step=false)

    @timeit timer "Assembly" begin
        # Residuals = collocation stress - imposed stress
        # Collocation stress
        @timeit timer "Collocation" mul!(solver.rhs, solver.E, solver.solution; global_index=true)
        # Fluid coupling
        if hasFluidCoupling(problem)
            @timeit timer "Fluid coupling" assembleFluidCouplingResidual!(solver, problem)
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

        # Reassemble HMatrix
        @timeit timer "H-matrix assembly" solver.mat = assemble_hmat(solver.K, solver.Xclt, solver.Xclt; solver.adm, solver.comp, global_index=true, threads=true, distributed=false)
    end
    return nothing
end

# Constraints
function assembleConstraintsResidualAndJacobian!(solver::DDSolver{R,T}, problem::NormalDDProblem{T}) where {R,T<:Real}
    # Loop over elements
    Threads.@threads for idx in eachindex(problem.mesh.elems)
        @inbounds (Res, Jac) = computeConstraints(problem.constraints, problem.w.value[idx] - problem.w.value_old[idx], problem.mesh.elems[idx].X)
        
        @inbounds solver.rhs[idx] -= Res
        @inbounds solver.mat_loc[1,1][idx] = -Jac
    end
    return nothing
end

function assembleConstraintsResidualAndJacobian!(solver::DDSolver{R,T}, problem::ShearDDProblem{T}) where {R,T<:Real}
    if (problem.n == problem.n_dof) # 1D mesh or 2D axis symmetric 
        # Loop over elements
        Threads.@threads for idx in eachindex(problem.mesh.elems)
            @inbounds (Res, Jac) = computeConstraints(problem.constraints[1], problem.δ.value[idx] - problem.δ.value_old[idx], problem.mesh.elems[idx].X)

            @inbounds solver.rhs[idx] -= Res
            @inbounds solver.mat_loc[1,1][idx] = -Jac
        end
    else
        # Loop over elements
        Threads.@threads for idx in eachindex(problem.mesh.elems)
            @inbounds (Res, Jac) = computeConstraints(problem.constraints[1], problem.δ.value[idx] - problem.δ.value_old[idx], problem.mesh.elems[idx].X)

            @inbounds solver.rhs[idx] -= Res
            @inbounds solver.mat_loc[1,1][idx] = -Jac

            @inbounds (Res, Jac) = computeConstraints(problem.constraints[2], problem.δ.value[problem.n + idx] - problem.δ.value_old[problem.n + idx], problem.mesh.elems[idx].X)

            @inbounds solver.rhs[problem.n + idx] -= Res
            @inbounds solver.mat_loc[2,2][idx] = -Jac
        end
    end
    return nothing
end

function assembleFluidCouplingResidual!(solver::DDSolver{R,T}, problem::NormalDDProblem{T}) where {R,T<:Real}
    Threads.@threads for idx in eachindex(problem.mesh.elems)
        @inbounds solver.rhs[idx] -= problem.fluid_coupling.p[idx] - problem.fluid_coupling.p_old[idx]
    end

    return nothing
end

function assembleFluidCouplingResidual!(solver::DDSolver{R,T}, problem::ShearDDProblem{T}) where {R,T<:Real}
    problem.σ.value = problem.σ.value_old
    problem.σ.value -= problem.fluid_coupling.p - problem.fluid_coupling.p_old
    return nothing
end


# Frictional constraints
function assembleFrictionResidualAndJacobian!(solver::DDSolver{R,T}, problem::ShearDDProblem{T}) where {R,T<:Real}
    if (problem.n == problem.n_dof) # 1D mesh or 2D axis symmetric
        # Loop over elements
        Threads.@threads for idx in eachindex(problem.mesh.elems)
            @inbounds (Res, Jac) = applyFrictionalConstraints(problem.friction, problem.δ.value[idx] - problem.δ.value_old[idx], problem.δ.value_old[idx], problem.σ.value[idx], problem.τ.value_old[idx])
            
            @inbounds solver.rhs[idx] -= Res
            @inbounds solver.mat_loc[1,1][idx] = -Jac
        end
    else
        # Loop over elements
        Threads.@threads for idx in eachindex(problem.mesh.elems)
            @inbounds (Res, Jac) = applyFrictionalConstraints(problem.friction, SVector(problem.δ.value[idx] - problem.δ.value_old[idx], problem.δ.value[problem.n+idx] - problem.δ.value_old[problem.n+idx]), SVector(problem.δ.value_old[idx], problem.δ.value_old[problem.n+idx]), problem.σ.value[idx], SVector(problem.τ.value_old[idx], problem.τ.value_old[problem.n+idx]))

            @inbounds solver.rhs[idx] -= Res[1]
            @inbounds solver.rhs[problem.n+idx] -= Res[2]
            @inbounds solver.mat_loc[1,1][idx] = -Jac[1,1]
            @inbounds solver.mat_loc[1,2][idx] = -Jac[1,2]
            @inbounds solver.mat_loc[2,1][idx] = -Jac[2,1]
            @inbounds solver.mat_loc[2,2][idx] = -Jac[2,2]
        end
    end
    return nothing
end

# Cohesive zone constraints
function assembleCohesiveZoneResidualAndJacobian!(solver::DDSolver{R,T}, problem::NormalDDProblem{T}) where {R,T<:Real}
    # Loop over elements
    Threads.@threads for idx in eachindex(problem.mesh.elems)
        @inbounds (Res, Jac) = applyCohesiveZoneConstraints(problem.cohesive, problem.w.value[idx] - problem.w.value_old[idx], problem.w.value_old[idx], problem.σ.value_old[idx], problem.mesh.elems[idx].X)

        @inbounds solver.rhs[idx] -= Res
        @inbounds solver.mat_loc[1,1][idx] = -Jac
    end
    return nothing
end

function assembleCohesiveZoneResidualAndJacobian!(solver::DDSolver{R,T}, problem::ShearDDProblem{T}) where {R,T<:Real}
    if (problem.n == problem.n_dof) # 1D mesh or 2D axis symmetric
        # Loop over elements
        Threads.@threads for idx in eachindex(problem.mesh.elems)
            @inbounds (Res, Jac) = applyCohesiveZoneConstraints(problem.cohesive, problem.δ.value[idx] - problem.δ.value_old[idx], problem.δ.value_old[idx], problem.τ.value_old[idx], problem.mesh.elems[idx].X)
            
            @inbounds solver.rhs[idx] -= Res
            @inbounds solver.mat_loc[1,1][idx] = -Jac
        end
    else
        # Loop over elements
        Threads.@threads for idx in eachindex(problem.mesh.elems)
            @inbounds (Res, Jac) = applyCohesiveZoneConstraints(problem.cohesive, SVector(problem.δ.value[idx] - problem.δ.value_old[idx], problem.δ.value[problem.n+idx] - problem.δ.value_old[problem.n+idx]), SVector(problem.δ.value_old[idx], problem.δ.value_old[problem.n+idx]), SVector(problem.τ.value_old[idx], problem.τ.value_old[problem.n+idx]), problem.mesh.elems[idx].X)

            @inbounds solver.rhs[idx] -= Res[1]
            @inbounds solver.rhs[problem.n+idx] -= Res[2]
            @inbounds solver.mat_loc[1,1][idx] = -Jac[1,1]
            @inbounds solver.mat_loc[1,2][idx] = -Jac[1,2]
            @inbounds solver.mat_loc[2,1][idx] = -Jac[2,1]
            @inbounds solver.mat_loc[2,2][idx] = -Jac[2,2]
        end
    end
    return nothing
end

function update!(problem::NormalDDProblem{T}, solver::DDSolver{R,T}) where {R,T<:Real}
    # Loop over elements
    Threads.@threads for idx in eachindex(problem.mesh.elems)
        @inbounds problem.w.value[idx] = problem.w.value_old[idx] + solver.solution[idx]
    end
    problem.σ.value = problem.σ.value_old + mul!(zeros(T, size(solver.solution)), solver.E, solver.solution; global_index=true, threads=false)
    # Fluid coupling
    if hasFluidCoupling(problem)
        problem.σ.value -= problem.fluid_coupling.p + problem.fluid_coupling.p_old
    end

    return nothing
end

function update!(problem::ShearDDProblem{T}, solver::DDSolver{R,T}) where {R,T<:Real}
    if (problem.n == problem.n_dof) # 1D mesh or 2D axis symmetric 
        # Loop over elements
        Threads.@threads for idx in eachindex(problem.mesh.elems)
            @inbounds problem.δ.value[idx] = problem.δ.value_old[idx] + solver.solution[idx]
        end
    else
        # Loop over elements
        Threads.@threads for idx in eachindex(problem.mesh.elems)
            @inbounds problem.δ.value[idx] = problem.δ.value_old[idx] + solver.solution[idx]
            @inbounds problem.δ.value[idx+problem.n] = problem.δ.value_old[idx+problem.n] + solver.solution[idx+problem.n]
        end
    end
    problem.σ.value = problem.σ.value_old
    problem.τ.value = problem.τ.value_old + mul!(zeros(T, size(solver.solution)), solver.E, solver.solution; global_index=true, threads=false)
    # Fluid coupling
    if hasFluidCoupling(problem)
        problem.σ.value -= problem.fluid_coupling.p - problem.fluid_coupling.p_old
    end

    return nothing
end

function computePressureCoupling!(problem::AbstractDDProblem{T}, time::T, timer::TimerOutput) where {T<:Real}
    # Check if problem has pressure coupling
    if hasFluidCoupling(problem)
        @timeit timer "Pressure update" begin
            updatePressure!(problem.fluid_coupling, [problem.mesh.elems[idx].X for idx in eachindex(problem.mesh.elems)], time)
        end
    end

    return nothing
end