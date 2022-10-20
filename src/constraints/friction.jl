abstract type AbstractFriction{T<:Real} end

function applyFrictionalConstraints(cst::AbstractFriction{T}, Δu::SVector{2, T}, t_old::SVector{2, T})::Tuple{SVector{2, T}, SVector{4, T}} where {T<:Real}
    # Plastic DD increment
    Δuᵖ = SVector(0.0, 0.0)
    # Compute trial tractions
    t_tr = t_old + computeTraction(cst, Δu, Δuᵖ)

    # Check yield conditions
    y = yieldFunction(cst, t_tr, 0.0)
    if (y <= 0.0)
        # Elastic update
        return (t_tr - t_old, computeTractionDerivative(cst, t_tr, Δu, Δuᵖ))
    else
        # Plastic update
        Δp = returnMap(cst, t_tr)
        # Rebuild plastic DD vector from scalar plastic DD
        Δuᵖ = reformPlasticDD(cst, Δp, t_tr)
        # Jacobian
        return (computeTraction(cst, Δu, Δuᵖ), computeTractionDerivative(cst, t_tr, Δu, Δuᵖ))
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
    for iter in 1:cst.max_iter
        Δp -= res / jac

        res = frictionResidual(cst, t_tr, Δp)
        jac = frictionJacobian(cst, t_tr, Δp)

        # Convergence check
        if ((abs(res) <= cst.abs_tol) || (abs(res / res_ini) <= cst.rel_tol))
            return Δp
        end
    end
    throw(ErrorException("Plastic update failed after $(iter) iterations!"))
end

# Default traction calculations
function computeTraction(cst::AbstractFriction{T}, Δu::SVector{2, T}, Δuᵖ::SVector{2, T})::Vector{T} where {T<:Real}
    return SVector(cst.kₙ * (Δu[1] - Δuᵖ[1]), cst.kₛ * (Δu[2] - Δuᵖ[2]))
end

function computeTractionDerivative(cst::AbstractFriction{T}, t_tr::SVector{2, T}, Δu::SVector{2, T}, Δuᵖ::SVector{2, T})::Vector{T} where {T<:Real}
    dΔuᵖ = computePlasticDDDerivative(cst, t_tr, Δuᵖ[2])
    return SVector(cst.kₙ * (1.0 - dΔuᵖ[1][1]), -cst.kₙ * dΔuᵖ[1][2], -cst.kₛ * dΔuᵖ[2][1], cst.kₛ * (1.0 - dΔuᵖ[2][2]))
end

function computeTraction(cst::AbstractFriction{T}, Δu::SVector{3, T}, Δuᵖ::SVector{3, T})::Vector{T} where {T<:Real}
    return SVector(cst.kₙ * (Δu[1] - Δuᵖ[1]), cst.kₛ * (Δu[2] - Δuᵖ[2]), cst.kₛ * (Δu[3] - Δuᵖ[3]))
end

# Default scalar to vector plastic DD
function reformPlasticDD(cst::AbstractFriction{T}, Δp::T, t_tr::SVector{2, T}) where {T<:Real}
    return SVector(0.0, Δp)
end

function reformPlasticDD(cst::AbstractFriction{T}, Δp::T, t_tr::SVector{3, T}) where {T<:Real}
    # Modify for 3D
    return SVector(0.0, Δp)
end

function plasticDDDerivative(cst::AbstractFriction{T}) where {T<:Real}
    return SVector(0.0, 1.0)
end

function computePlasticDDDerivative(cst::AbstractFriction{T}, t_tr::SVector{2, T}, Δp::T) where {T<:Real}
    df_dt = yieldFunctionStressDerivative(cst, t_tr, Δp)
    df_dp = yieldFunctionDerivative(cst, t_tr, Δp)
    dΔuᵖ_dΔp = plasticDDDerivative(cst)
    if Δp > 0.0
        return SVector(SVector(-dΔuᵖ_dΔp[1] * cst.kₙ * df_dt[1] / df_dp, -dΔuᵖ_dΔp[1] * cst.kₛ * df_dt[2] / df_dp), SVector(-dΔuᵖ_dΔp[2] * cst.kₙ * df_dt[1] / df_dp, -dΔuᵖ_dΔp[2] * cst.kₛ * df_dt[2] / df_dp))
    else
        return return SVector(SVector(0.0, 0.0), SVector(0.0, 0.0))
    end
end

# Default residual and jacobian (plasticity)
function frictionResidual(cst::AbstractFriction{T}, t_tr::SVector{2, T}, Δp::T) where {T<:Real}
    return yieldFunction(cst, t_tr, Δp)
end

function frictionJacobian(cst::AbstractFriction{T}, t_tr::SVector{2, T}, Δp::T) where {T<:Real}
    return yieldFunctionDerivative(cst, t_tr, Δp)
end

function frictionResidual(cst::AbstractFriction{T}, t_tr::SVector{3, T}, Δp::T) where {T<:Real}
    return yieldFunction(cst, t_tr, Δp)
end

function frictionJacobian(cst::AbstractFriction{T}, t_tr::SVector{3, T}, Δp::T) where {T<:Real}
    return yieldFunctionDerivative(cst, t_tr, Δp)
end

# To replace in children classes
function yieldFunction(cst::AbstractFriction{T}, t_tr::SVector{2, T}, Δp::T) where {T<:Real}
    throw(MethodError(yieldFunction, cst))
end

function yieldFunctionDerivative(cst::AbstractFriction{T}, t_tr::SVector{2, T}, Δp::T) where {T<:Real}
    throw(MethodError(yieldFunctionDerivative, cst))
end

function yieldFunctionStressDerivative(cst::AbstractFriction{T}, t_tr::SVector{2, T}, Δp::T) where {T<:Real}
    throw(MethodError(yieldFunctionStressDerivative, cst))
end

# Constant yield model
struct ConstantYield{T<:Real} <: AbstractFriction{T}
    " Shear stress yield"
    τ₀::T

    " Normal stiffness"
    kₙ::T

    " Shear stiffness"
    kₛ::T

    " Absolute tolerance"
    abs_tol::T

    " Relative tolerance"
    rel_tol::T

    " Maximum iterations"
    max_iter::Int

    " Constructor"
    function ConstantYield(τ₀::T, kₙ::T, kₛ::T; abs_tol::T = 1.0e-12, rel_tol::T = 1.0e-10, max_iter::Int = 20) where {T<:Real}
        return new{T}(τ₀, kₙ, kₛ, abs_tol, rel_tol, max_iter)
    end
end

function plasticDDDerivative(cst::ConstantYield{T}) where {T<:Real}
    return SVector(0.0, 1.0)
end

function yieldFunction(cst::ConstantYield{T}, t_tr::SVector{2, T}, Δp::T) where {T<:Real}
    return (t_tr[2] - cst.kₛ * Δp) - cst.τ₀
end

function yieldFunctionDerivative(cst::ConstantYield{T}, t_tr::SVector{2, T}, Δp::T) where {T<:Real}
    return - cst.kₛ
end

function yieldFunctionStressDerivative(cst::ConstantYield{T}, t_tr::SVector{2, T}, Δp::T) where {T<:Real}
    return SVector(0.0, 1.0)
end

function reformPlasticDD(cst::ConstantYield{T}, Δp::T, t_tr::SVector{2, T}) where {T<:Real}
    return SVector(0.0, Δp)
end

function reformPlasticDD(cst::ConstantYield{T}, Δp::T, t_tr::SVector{3, T}) where {T<:Real}
    # Modify for 3D
    return SVector(0.0, Δp)
end

# Constant friction model
struct ConstantFriction{T<:Real} <: AbstractFriction{T}
    " Friction coefficient"
    f::T

    " Dilation coefficient"
    ζ::T

    " Normal stiffness"
    kₙ::T

    " Shear stiffness"
    kₛ::T

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

function plasticDDDerivative(cst::ConstantFriction{T}) where {T<:Real}
    return SVector(-cst.ζ, 1.0)
end

function yieldFunction(cst::ConstantFriction{T}, t_tr::SVector{2, T}, Δp::T) where {T<:Real}
    return (t_tr[2] - cst.kₛ * Δp) - cst.f * (t_tr[1] + cst.kₙ * cst.ζ * Δp)
end

function yieldFunctionDerivative(cst::ConstantFriction{T}, t_tr::SVector{2, T}, Δp::T) where {T<:Real}
    return - cst.kₛ - cst.f * cst.ζ * cst.kₙ
end

function yieldFunctionStressDerivative(cst::ConstantFriction{T}, t_tr::SVector{2, T}, Δp::T) where {T<:Real}
    return SVector(-cst.f, 1.0)
end

function reformPlasticDD(cst::ConstantFriction{T}, Δp::T, t_tr::SVector{2, T}) where {T<:Real}
    return SVector(-cst.ζ * Δp, Δp)
end

function reformPlasticDD(cst::ConstantFriction{T}, Δp::T, t_tr::SVector{3, T}) where {T<:Real}
    # Modify for 3D
    return SVector(-cst.ζ * Δp, Δp)
end