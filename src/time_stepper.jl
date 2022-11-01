abstract type AbstractTimeStepper{T<:Real} end

mutable struct TimeSequence{T<:Real} <: AbstractTimeStepper{T}
    " Start time"
    start_time::T

    " End time"
    end_time::T

    " Time sequence"
    time_seq::Vector{T}

    " Tolerance"
    tol::T

    " Constructor"
    function TimeSequence(time_seq::Vector{T}; start_time::T=time_seq[1], end_time::T=time_seq[end], tol::T=1.0e-08) where {T<:Real}
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

        return new{T}(start_time, end_time, time_seq, tol)
    end
end

function get_time(ts::TimeSequence{T}, it::Int) where {T<:Real}
    return ts.time_seq[it+1]
end
