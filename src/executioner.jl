abstract type AbstractExecutioner{T<:Real} end

mutable struct SteadyExecutioner{T<:Real} <: AbstractExecutioner{T}
    " Time step number"
    time_step::Int

    " Current time"
    time::T

    " Old time"
    time_old::T

    " Current time step size"
    dt::T

    " Solve converged?"
    converged::Bool

    " Constructor"
    function SteadyExecutioner(T::DataType)
        return new{T}(0, 0.0, 0.0, 0.0, true)
    end
end

function advanceTime!(exec::SteadyExecutioner{T}) where {T<:Real}
    # Update time step
    exec.time_step = exec.time_step + 1
    # Update time old
    exec.time_old = exec.time
    # Update time
    exec.time = 1.0
    # Update dt
    exec.dt = exec.time - exec.time_old

    return nothing
end

mutable struct TransientExecutioner{T<:Real} <: AbstractExecutioner{T}
    " Time step number"
    time_step::Int

    " Current time"
    time::T

    " Old time"
    time_old::T

    " Current time step size"
    dt::T

    " Solve converged?"
    converged::Bool

    " Constructor"
    function TransientExecutioner(time_stepper::AbstractTimeStepper{T}) where {T<:Real}
        return new{T}(0, time_stepper.start_time, time_stepper.start_time, 0.0, true)
    end
end

function advanceTime!(exec::TransientExecutioner{T}, time_stepper::AbstractTimeStepper{T}) where {T<:Real}
    if exec.converged
        # Update time step
        exec.time_step = exec.time_step + 1
        # Update time old
        exec.time_old = exec.time
        # Update time step
        exec.dt = getCurrentTimeStep(time_stepper, exec.time_old, exec.dt, exec.time_step)
    else
        # Update time step
        exec.dt = cutBackTimeStep(time_stepper, exec.dt)
    end

    # Update time
    exec.time = exec.time_old + exec.dt

    return nothing
end

function printTimeStepInfo(exec::TransientExecutioner{T}) where {T<:Real}
    @printf("\nTime Step %i, time = %.6f, dt = %.6f\n", exec.time_step, exec.time, exec.dt)
    return nothing
end
