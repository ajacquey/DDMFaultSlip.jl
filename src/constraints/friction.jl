abstract type AbstractFriction{T<:Real} end

function applyFrictionalConstraints!(cst::AbstractFriction{T}, Δu::SVector{2, T}, t_old::SVector{2, T}) where {T<:Real}
    # Compute trial tractions
    t_tr = t_old + computeTrialTraction(cst, Δu, SVector(0.0, 0.0))

    # Check yield conditions
    y = yieldFunction(cst, t_tr, 0.0)
    if (y <= 0.0)
        # Elastic update
        return SVector(0.0, 0.0)
    else
        # Plastic update
        Δp = returnMap(cst, t_tr)
        # Rebuild plastic DD vector from scalar plastic DD
        Δuᵖ = reformPlasticDD(cst, Δp, t_tr)
        return computeTrialTraction(cst, Δu, Δuᵖ)
    end
end

function returnMap(cst::AbstractFriction{T}, t_tr::SVector{2, T})::T where {T<:Real}
    # Initialized scalar plastic DD
    Δp = 0.0

    # Initial residual
    res_ini = frictionResidual(cst, t_tr, Δp)

    res = res_ini
    jac = frictionJacobian(cst, t_tr, Δp)

    # Newton loop
    for _ in 1:max_iter
        Δp -= res / jac

        res = frictionResidual(cst, t_tr, Δp)
        jac = frictionJacobian(cst, t_tr, Δp)

        # Convergence check
        if ((abs(res) <= abs_tol) || (abs(res / res_ini) <= rel_tol))
            return Δp
        end
    end
    throw(ErrorException("Plastic update failed after $(iter) iterations!"))
end

function computeTrialTraction(cst::AbstractFriction{T}, Δu::SVector{2, T}, Δuᵖ::SVector{2, T})::Vector{T} where {T<:Real}
    return SVector(cst.kₙ * (Δu[1] - Δuᵖ[1]), cst.kₛ * (Δu[2] - Δuᵖ[2]))
end

function computeTrialTraction(cst::AbstractFriction{T}, Δu::SVector{3, T}, Δuᵖ::SVector{3, T})::Vector{T} where {T<:Real}
    return SVector(cst.kₙ * (Δu[1] - Δuᵖ[1]), cst.kₛ * (Δu[2] - Δuᵖ[2]), cst.kₛ * (Δu[3] - Δuᵖ[3]))
end

function reformPlasticDD(cst::AbstractFriction{T}, Δp::T, t_tr::SVector{2, T}) where {T<:Real}
    return SVector(-cst.zeta * Δp, Δp)
end

function reformPlasticDD(cst::AbstractFriction{T}, Δp::T, t_tr::SVector{3, T}) where {T<:Real}
    # Modify for 3D
    return SVector(-cst.zeta * Δp, Δp)
end

function yieldFunction(cst::AbstractFriction{T}, t_tr::SVector{2, T}, Δp::T) where {T<:Real}
    throw(MethodError(yieldFunction, cst))
end

function yieldFunctionDerivative(cst::AbstractFriction{T}, t_tr::SVector{2, T}, Δp::T) where {T<:Real}
    throw(MethodError(yieldFunctionDerivative, cst))
end

struct ConstantFriction{T<:Real} <: AbstractFriction{T}
    " Friction coefficient"
    f::T

    " Dilation coefficient"
    ζ::T

    " Shear stiffness"
    kₛ::T

    " Normal stiffness"
    kₙ::T

    " Absolute tolerance"
    abs_tol::T

    " Relative tolerance"
    rel_tol::T

    " Maximum iterations"
    max_iter::Int

    " Constructor"
    function ConstantFriction(f::T, kₙ::T, kₛ::T; ζ::T = 0.0, abs_tol::T = 1.0e-12, rel_tol::T = 1.0e-10, max_iter::Int = 20) where {T<:Real}
        return new{T}(f, ζ, kₙ, kₛ, abs_tol, rel_tol, max_iter)
    end
end

function frictionResidual(cst::ConstantFriction{T}, t_tr::SVector{2, T}, Δp::T) where {T<:Real}
    return yieldFunction(cst, t_tr, Δp)
end

function frictionJacobian(cst::ConstantFriction{T}, t_tr::SVector{2, T}, Δp::T) where {T<:Real}
    return yieldFunctionDerivative(cst, t_tr, Δp)
end

function yieldFunction(cst::ConstantFriction{T}, t_tr::SVector{2, T}, Δp::T) where {T<:Real}
    return (t_tr[2] - cst.kₛ * Δp) - cst.f * (t_tr[1] - cst.kₙ * cst.ζ * Δp)
end

function yieldFunctionDerivative(cst::ConstantFriction{T}, t_tr::SVector{2, T}, Δp::T) where {T<:Real}
    return - cst.kₛ + cst.f * cst.ζ * cst.kₙ
end