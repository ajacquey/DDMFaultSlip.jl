abstract type AbstractFriction{T<:Real} end

function applyFrictionalConstraints(cst::AbstractFriction{T}, Δu::SVector{N,T}, u_old::SVector{N,T}, t_old::SVector{N,T}) where {N,T<:Real}
    # Plastic DD increment
    Δuᵖ = @SVector zeros(T, N)
    # Compute trial tractions
    t_tr = t_old + computeTraction(cst, Δu, Δuᵖ)
    # Compute scalar tractions
    (σ_tr, τ_tr) = computeScalarTraction(cst, t_tr)
    # Pre return map update
    preReturnMap(cst, u_old, Δu)

    # Check yield conditions
    y = yieldFunction(cst, σ_tr, τ_tr, 0.0)
    if (y <= 0.0)
        # Elastic update
        return (t_tr - t_old, computeTractionDerivative(cst, t_tr, Δu, Δuᵖ))
    else
        # Plastic update
        Δp = returnMap(cst, σ_tr, τ_tr)
        # Rebuild plastic DD vector from scalar plastic DD
        Δuᵖ = reformPlasticDD(cst, Δp, t_tr)
        # Post return map update
        postReturnMap(cst, Δp)
        # Jacobian
        return (computeTraction(cst, Δu, Δuᵖ), computeTractionDerivative(cst, t_tr, Δu, Δuᵖ))
    end
end

function returnMap(cst::AbstractFriction{T}, σ_tr::T, τ_tr::T)::T where {T<:Real}
    # Initialized scalar plastic DD
    Δp = 0.0

    # Initial residual
    res_ini = frictionResidual(cst, σ_tr, τ_tr, Δp)

    res = res_ini
    jac = frictionJacobian(cst, σ_tr, τ_tr, Δp)

    # Newton loop
    for iter in 1:cst.max_iter
        Δp -= res / jac

        res = frictionResidual(cst, σ_tr, τ_tr, Δp)
        jac = frictionJacobian(cst, σ_tr, τ_tr, Δp)

        # Convergence check
        if ((abs(res) <= cst.abs_tol) || (abs(res / res_ini) <= cst.rel_tol))
            return Δp
        end
    end
    throw(ErrorException("Plastic update failed after $(iter) iterations!"))
end

# Default traction calculations - 2D
function computeTraction(cst::AbstractFriction{T}, Δu::SVector{2,T}, Δuᵖ::SVector{2,T})::SVector{2, T} where {T<:Real}
    return SVector(cst.kₙ * (Δu[1] - Δuᵖ[1]), cst.kₛ * (Δu[2] - Δuᵖ[2]))
end

function computeTractionDerivative(cst::AbstractFriction{T}, t_tr::SVector{2,T}, Δu::SVector{2,T}, Δuᵖ::SVector{2,T})::SMatrix{2,2,T} where {T<:Real}
    dΔuᵖ = computePlasticDDDerivative(cst, t_tr, computeScalarPlasticDD(cst, Δuᵖ))
    return SMatrix{2}(cst.kₙ * (1.0 - dΔuᵖ[1, 1]), -cst.kₙ * dΔuᵖ[2, 1], -cst.kₛ * dΔuᵖ[1, 2], cst.kₛ * (1.0 - dΔuᵖ[2, 2]))
end

# Default traction calculations - 3D
function computeTraction(cst::AbstractFriction{T}, Δu::SVector{3,T}, Δuᵖ::SVector{3,T})::SVector{3,T} where {T<:Real}
    return SVector(cst.kₙ * (Δu[1] - Δuᵖ[1]), cst.kₛ * (Δu[2] - Δuᵖ[2]), cst.kₛ * (Δu[3] - Δuᵖ[3]))
end

function computeTractionDerivative(cst::AbstractFriction{T}, t_tr::SVector{3,T}, Δu::SVector{3,T}, Δuᵖ::SVector{3,T})::SMatrix{3,3,T} where {T<:Real}
    dΔuᵖ = computePlasticDDDerivative(cst, t_tr, computeScalarPlasticDD(cst, Δuᵖ))
    return SMatrix{3}(cst.kₙ * (1.0 - dΔuᵖ[1, 1]), -cst.kₙ * dΔuᵖ[2, 1], -cst.kₙ * dΔuᵖ[3, 1], -cst.kₛ * dΔuᵖ[1, 2], cst.kₛ * (1.0 - dΔuᵖ[2, 2]), -cst.kₛ * dΔuᵖ[3, 2], -cst.kₛ * dΔuᵖ[1, 3], -cst.kₛ * dΔuᵖ[2, 3], cst.kₛ * (1.0 - dΔuᵖ[3, 3]))
end

# Default scalar traction calculations - 2D
function computeScalarTraction(cst::AbstractFriction{T}, t_tr::SVector{2,T})::Tuple{T,T} where {T<:Real}
    return t_tr[1], t_tr[2]
end

# Default scalar traction calculations - 3D
function computeScalarTraction(cst::AbstractFriction{T}, t_tr::SVector{3,T})::Tuple{T,T} where {T<:Real}
    return t_tr[1], sqrt(t_tr[2]^2 + t_tr[3]^2)
end

# Default scalar plastic slip increment calculations - 2D
function computeScalarPlasticDD(cst::AbstractFriction{T}, Δuᵖ::SVector{2,T})::T where {T<:Real}
    return Δuᵖ[2]
end

# Default scalar plastic slip increment calculations - 3D
function computeScalarPlasticDD(cst::AbstractFriction{T}, Δuᵖ::SVector{3,T})::T where {T<:Real}
    return sqrt(Δuᵖ[2]^2 + Δuᵖ[3]^2)
end

# Default scalar to vector plastic DD
function reformPlasticDD(cst::AbstractFriction{T}, Δp::T, t_tr::SVector{2,T})::SVector{2,T} where {T<:Real}
    return SVector{2}(0.0, Δp)
end

function reformPlasticDD(cst::AbstractFriction{T}, Δp::T, t_tr::SVector{3,T})::SVector{3,T} where {T<:Real}
    (σ_tr, τ_tr) = computeScalarTraction(cst, t_tr)
    return SVector{3}(0.0, Δp * t_tr[2] / τ_tr, Δp * t_tr[3] / τ_tr)
end

# Default plastic flow direction - 2D
function plasticFlowDirection(cst::AbstractFriction{T}, t_tr::SVector{2,T})::SVector{2,T} where {T<:Real}
    return SVector{2}(0.0, 1.0)
end

function plasticFlowDirectionDerivative(cst::AbstractFriction{T}, t_tr::SVector{2,T})::SMatrix{2,2,T} where {T<:Real}
    return SMatrix{2}(0.0, 0.0, 0.0, 0.0)
end

# Default plastic flow direction - 3D
function plasticFlowDirection(cst::AbstractFriction{T}, t_tr::SVector{3,T})::SVector{3,T} where {T<:Real}
    (σ_tr, τ_tr) = computeScalarTraction(cst, t_tr)
    return SVector{3}(0.0, t_tr[2] / τ_tr, t_tr[3] / τ_tr)
end

function plasticFlowDirectionDerivative(cst::AbstractFriction{T}, t_tr::SVector{3,T})::SMatrix{3,3,T} where {T<:Real}
    (σ_tr, τ_tr) = computeScalarTraction(cst, t_tr)
    return SMatrix{3}(0.0, 0.0, 0.0, 0.0, cst.kₛ / τ_tr * (1.0 - (t_tr[2] / τ_tr)^2), -cst.kₛ / τ_tr * t_tr[2] * t_tr[3] / τ_tr^2, 0.0, -cst.kₛ / τ_tr * t_tr[2] * t_tr[3] / τ_tr^2, cst.kₛ / τ_tr * (1.0 - (t_tr[3] / τ_tr)^2))
end

function plasticMultiplierDerivative(cst::AbstractFriction{T}, t_tr::SVector{2,T}, Δp::T)::SVector{2,T} where {T<:Real}
    (σ_tr, τ_tr) = computeScalarTraction(cst, t_tr)
    df_dt = yieldFunctionStressDerivative(cst, σ_tr, τ_tr, Δp)
    df_dΔp = yieldFunctionDerivative(cst, σ_tr, τ_tr, Δp)
    return SVector{2}(-cst.kₙ * df_dt[1] / df_dΔp, -cst.kₛ * df_dt[2] / df_dΔp)
end

function plasticMultiplierDerivative(cst::AbstractFriction{T}, t_tr::SVector{3,T}, Δp::T)::SVector{3,T} where {T<:Real}
    (σ_tr, τ_tr) = computeScalarTraction(cst, t_tr)
    df_dt = yieldFunctionStressDerivative(cst, σ_tr, τ_tr, Δp)
    df_dΔp = yieldFunctionDerivative(cst, σ_tr, τ_tr, Δp)
    return SVector{3}(-cst.kₙ * df_dt[1] / df_dΔp, -cst.kₛ * df_dt[2] * t_tr[2] / τ_tr / df_dΔp, -cst.kₛ * df_dt[2] * t_tr[3] / τ_tr / df_dΔp)
end

function computePlasticDDDerivative(cst::AbstractFriction{T}, t_tr::SVector{2,T}, Δp::T)::SMatrix{2,2,T} where {T<:Real}
    r = plasticFlowDirection(cst, t_tr)
    dΔp_dΔu = plasticMultiplierDerivative(cst, t_tr, Δp)
    dr_dΔu = plasticFlowDirectionDerivative(cst, t_tr)

    if Δp > 0.0
        return SMatrix{2}(dΔp_dΔu[1] * r[1] + Δp * dr_dΔu[1,1], dΔp_dΔu[2] * r[1] + Δp * dr_dΔu[2,1], 
            dΔp_dΔu[1] * r[2] + Δp * dr_dΔu[1,2], dΔp_dΔu[2] * r[2] + Δp * dr_dΔu[2,2])
    else
        return SMatrix{2}(0.0, 0.0, 0.0, 0.0)
    end
end

function computePlasticDDDerivative(cst::AbstractFriction{T}, t_tr::SVector{3,T}, Δp::T)::SMatrix{3,3,T} where {T<:Real}
    r = plasticFlowDirection(cst, t_tr)
    dΔp_dΔu = plasticMultiplierDerivative(cst, t_tr, Δp)
    dr_dΔu = plasticFlowDirectionDerivative(cst, t_tr)
    if Δp > 0.0
        return SMatrix{3}(dΔp_dΔu[1] * r[1] + Δp * dr_dΔu[1,1], dΔp_dΔu[2] * r[1] + Δp * dr_dΔu[2,1], dΔp_dΔu[3] * r[1] + Δp * dr_dΔu[3,1],
            dΔp_dΔu[1] * r[2] + Δp * dr_dΔu[1,2], dΔp_dΔu[2] * r[2] + Δp * dr_dΔu[2,2], dΔp_dΔu[3] * r[2] + Δp * dr_dΔu[3,2],
            dΔp_dΔu[1] * r[3] + Δp * dr_dΔu[1,3], dΔp_dΔu[2] * r[3] + Δp * dr_dΔu[2,3], dΔp_dΔu[3] * r[3] + Δp * dr_dΔu[3,3])
    else
        return SMatrix{3}(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    end
end

# Default residual and jacobian (plasticity)
function frictionResidual(cst::AbstractFriction{T}, σ_tr::T, τ_tr::T, Δp::T)::T where {T<:Real}
    return yieldFunction(cst, σ_tr, τ_tr, Δp)
end

function frictionJacobian(cst::AbstractFriction{T}, σ_tr::T, τ_tr::T, Δp::T)::T where {T<:Real}
    return yieldFunctionDerivative(cst, σ_tr, τ_tr, Δp)
end

# To replace in children classes
function yieldFunction(cst::AbstractFriction{T}, σ_tr::T, τ_tr::T, Δp::T) where {T<:Real}
    throw(MethodError(yieldFunction, cst))
end

function yieldFunctionDerivative(cst::AbstractFriction{T}, σ_tr::T, τ_tr::T, Δp::T) where {T<:Real}
    throw(MethodError(yieldFunctionDerivative, cst))
end

function yieldFunctionStressDerivative(cst::AbstractFriction{T}, σ_tr::T, τ_tr::T, Δp::T) where {T<:Real}
    throw(MethodError(yieldFunctionStressDerivative, cst))
end

function preReturnMap(cst::AbstractFriction{T}, u_old::SVector{N,T}, Δu::SVector{N,T}) where {N,T<:Real}
    return nothing
end

function postReturnMap(cst::AbstractFriction{T}, Δp::T) where {T<:Real}
    return nothing
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
    function ConstantYield(τ₀::T, kₙ::T, kₛ::T; abs_tol::T=1.0e-12, rel_tol::T=1.0e-10, max_iter::Int=20) where {T<:Real}
        return new{T}(τ₀, kₙ, kₛ, abs_tol, rel_tol, max_iter)
    end
end

function yieldFunction(cst::ConstantYield{T}, σ_tr::T, τ_tr::T, Δp::T)::T where {T<:Real}
    return (τ_tr - cst.kₛ * Δp) - cst.τ₀
end

function yieldFunctionDerivative(cst::ConstantYield{T}, σ_tr::T, τ_tr::T, Δp::T)::T where {T<:Real}
    return -cst.kₛ
end

function yieldFunctionStressDerivative(cst::ConstantYield{T}, σ_tr::T, τ_tr::T, Δp::T)::SVector{2,T} where {T<:Real}
    return SVector{2}(0.0, 1.0)
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
    function ConstantFriction(f::T, kₙ::T, kₛ::T; ζ::T=0.0, abs_tol::T=1.0e-12, rel_tol::T=1.0e-10, max_iter::Int=20) where {T<:Real}
        return new{T}(f, ζ, kₙ, kₛ, abs_tol, rel_tol, max_iter)
    end
end

function plasticFlowDirection(cst::ConstantFriction{T}, t_tr::SVector{2,T})::SVector{2,T} where {T<:Real}
    return SVector{2}(-cst.ζ, 1.0)
end

function plasticFlowDirection(cst::ConstantFriction{T}, t_tr::SVector{3,T})::SVector{3,T} where {T<:Real}
    (σ_tr, τ_tr) = computeScalarTraction(cst, t_tr)
    return SVector{3}(-cst.ζ, t_tr[2] / τ_tr, t_tr[3] / τ_tr)
end

function yieldFunction(cst::ConstantFriction{T}, σ_tr::T, τ_tr::T, Δp::T)::T where {T<:Real}
    return (τ_tr - cst.kₛ * Δp) - cst.f * (σ_tr + cst.kₙ * cst.ζ * Δp)
end

function yieldFunctionDerivative(cst::ConstantFriction{T}, σ_tr::T, τ_tr::T, Δp::T)::T where {T<:Real}
    return -cst.kₛ - cst.f * cst.ζ * cst.kₙ
end

function yieldFunctionStressDerivative(cst::ConstantFriction{T}, σ_tr::T, τ_tr::T, Δp::T)::SVector{2,T} where {T<:Real}
    return SVector{2}(-cst.f, 1.0)
end

function reformPlasticDD(cst::ConstantFriction{T}, Δp::T, t_tr::SVector{2,T})::SVector{2,T} where {T<:Real}
    return Δp * plasticFlowDirection(cst, t_tr)
end

function reformPlasticDD(cst::ConstantFriction{T}, Δp::T, t_tr::SVector{3,T})::SVector{3,T} where {T<:Real}
    return Δp * plasticFlowDirection(cst, t_tr)
end

# Constant friction model
mutable struct SlipWeakeningFriction{T<:Real} <: AbstractFriction{T}
    " Old value of slip"
    δ_old::T

    " Peak friction coefficient"
    fₚ::T

    " Residual friction coefficient"
    fᵣ::T

    " Residual slip"
    δᵣ::T

    " Slope of weakening"
    w::T
    
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
    function SlipWeakeningFriction(fₚ::T, fᵣ::T, δᵣ::T, kₙ::T, kₛ::T; ζ::T=0.0, abs_tol::T=1.0e-12, rel_tol::T=1.0e-10, max_iter::Int=20) where {T<:Real}
        return new{T}(0.0, fₚ, fᵣ, δᵣ, (fₚ - fᵣ) / δᵣ, ζ, kₙ, kₛ, abs_tol, rel_tol, max_iter)
    end
end

function sigmund(x::T)::T where {T<:Real}
    return 1.0 / (1.0 + exp(-x))
end

function friction(cst::SlipWeakeningFriction{T}, Δp::T)::T where {T<:Real}
    if ((cst.δ_old + Δp) < cst.δᵣ)
        return cst.fₚ - cst.w * (cst.δ_old + Δp)
    else
        return cst.fᵣ
    end
end

function frictionSlipDerivative(cst::SlipWeakeningFriction{T}, Δp::T)::T where {T<:Real}
    # Here we use a smooth derivative for convergence
    k = 50.0 / cst.δᵣ
    return cst.w * (sigmund(k * (cst.δ_old + Δp - cst.δᵣ)) - 1.0)
end

function plasticFlowDirection(cst::SlipWeakeningFriction{T}, t_tr::SVector{2,T})::SVector{2,T} where {T<:Real}
    return SVector{2}(-cst.ζ, 1.0)
end

function plasticFlowDirection(cst::SlipWeakeningFriction{T}, t_tr::SVector{3,T})::SVector{3,T} where {T<:Real}
    (σ_tr, τ_tr) = computeScalarTraction(cst, t_tr)
    return SVector{3}(-cst.ζ, t_tr[2] / τ_tr, t_tr[3] / τ_tr)
end

function yieldFunction(cst::SlipWeakeningFriction{T}, σ_tr::T, τ_tr::T, Δp::T)::T where {T<:Real}
    f = friction(cst, Δp)
    return (τ_tr - cst.kₛ * Δp) - f * (σ_tr + cst.kₙ * cst.ζ * Δp)
end

function yieldFunctionDerivative(cst::SlipWeakeningFriction{T}, σ_tr::T, τ_tr::T, Δp::T)::T where {T<:Real}
    f = friction(cst, Δp)
    df = frictionSlipDerivative(cst, Δp)
    return -cst.kₛ - f * cst.ζ * cst.kₙ -df * (σ_tr + cst.kₙ * cst.ζ * Δp)
end

function yieldFunctionStressDerivative(cst::SlipWeakeningFriction{T}, σ_tr::T, τ_tr::T, Δp::T)::SVector{2,T} where {T<:Real}
    f = friction(cst, Δp)
    return SVector{2}(-f, 1.0)
end

function reformPlasticDD(cst::SlipWeakeningFriction{T}, Δp::T, t_tr::SVector{2,T})::SVector{2,T} where {T<:Real}
    return Δp * plasticFlowDirection(cst, t_tr)
end

function reformPlasticDD(cst::SlipWeakeningFriction{T}, Δp::T, t_tr::SVector{3,T})::SVector{3,T} where {T<:Real}
    return Δp * plasticFlowDirection(cst, t_tr)
end

function preReturnMap(cst::SlipWeakeningFriction{T}, u_old::SVector{N,T}, Δu::SVector{N,T}) where {N,T<:Real}
    # Compute scalar old DD
    cst.δ_old = computeScalarPlasticDD(cst, u_old)
    return nothing
end

function postReturnMap(cst::SlipWeakeningFriction{T}, Δp::T) where {T<:Real}
    return nothing
end