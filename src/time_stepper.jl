abstract type AbstractTimeStepper{T<:Real} end

function cutBackTimeStep(ts::AbstractTimeStepper{T}, dt::T) where {T<:Real}
    return ts.cutback_factor * dt
end

struct ConstantDT{T<:Real} <: AbstractTimeStepper{T}
    " Start time"
    start_time::T

    " End time"
    end_time::T

    " Time step size"
    dt::T

    " Growth factor after a successful solve"
    growth_factor::T

    " Cutback factor after a failed solve"
    cutback_factor::T

    " Tolerance"
    tol::T

    " Constructor"
    function ConstantDT(start_time::T, end_time::T, n::Int; growth_factor::T=2.0, cutback_factor::T=0.5, tol::T=1.0e-08) where {T<:Real}
        dt = (end_time - start_time) / n
        return new{T}(start_time, end_time, dt, growth_factor, cutback_factor, tol)
    end
end

function getCurrentTimeStep(ts::ConstantDT{T}, t_old::T, dt_old::T, time_step::Int) where {T<:Real}
    if time_step < 2
        return ts.dt
    end
    dt = min(ts.growth_factor * dt_old, ts.dt)
    if t_old + dt > ts.end_time
        dt = ts.end_time - t_old
    end

    return dt
end

struct TimeSequence{T<:Real} <: AbstractTimeStepper{T}
    " Start time"
    start_time::T

    " End time"
    end_time::T

    " Time sequence"
    time_seq::Vector{T}

    " Cutback factor after a failed solve"
    cutback_factor::T

    " Tolerance"
    tol::T

    " Constructor"
    function TimeSequence(time_seq::Vector{T}; start_time::T=time_seq[1], end_time::T=time_seq[end], cutback_factor::T=0.5, tol::T=1.0e-08) where {T<:Real}
        if (~issorted(time_seq))
            throw(DomainError(time_seq, "Time sequence need to be sorted!"))
        end

        # Check that starting time is either first elem of lower
        if (start_time > time_seq[1])
            throw(DomainError(start_time, "Start time should be smaller or equal than first time in sequence!"))
        elseif (start_time == time_seq[1] != 0.0)
            pushfirst!(time_seq, 0.0)
        elseif ((start_time == 0.0) && (time_seq[1] != 0.0))
            pushfirst!(time_seq, 0.0)
        end

        # Check that end time is either last elem or bigger
        if (end_time < time_seq[end])
            throw(DomainError(end_time, "End time should be bigger or equal than last time in sequence!"))
        elseif (end_time > time_seq[end])
            push!(time_seq, end_time)
        end

        return new{T}(start_time, end_time, time_seq, cutback_factor, tol)
    end
end

function getCurrentTimeStep(ts::TimeSequence{T}, t_old::T, dt_old::T, time_step::Int) where {T<:Real}
    dt = ts.time_seq[time_step+1] -ts.time_seq[time_step]
    if t_old + dt > ts.end_time
        dt = ts.end_time - t_old
    end

    return dt
end