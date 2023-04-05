abstract type AbstractFriction end

struct DefaultFriction <: AbstractFriction
end

function applyFrictionalConstraints(cst::AbstractFriction, Δδ::T, δ_old::T, σ::T, τ_old::T) where {N,T<:Real}
    ErrorException("HERE!")
    # Plastic DD slip
    Δδᵖ = 0.0
    # Compute trial shear stress
    τ_tr = τ_old + computeTraction(cst, Δδ, Δδᵖ)
    # Pre return map update
    preReturnMap(cst, δ_old, Δδ)
    
    # Check yield conditions
    y = yieldFunction(cst, σ, τ_tr, 0.0)
    if (y <= 0.0)
        # Elastic update
        return (τ_tr - τ_old, computeTractionDerivative(cst, σ, τ_tr, 0.0))
    else
        # Plastic update
        Δδᵖ = returnMap(cst, σ, τ_tr)
        # Post return map update
        postReturnMap(cst, Δδᵖ)
        # Jacobian
        return (computeTraction(cst, Δδ, Δδᵖ), computeTractionDerivative(cst, σ, τ_tr, Δδᵖ))
    end
end

function applyFrictionalConstraints(cst::AbstractFriction, Δu::SVector{N,T}, u_old::SVector{N,T}, σ::T, t_old::SVector{N,T}) where {N,T<:Real}
    # Plastic DD increment
    Δuᵖ = @SVector zeros(T, N)
    # Compute trial tractions
    t_tr = t_old + computeTraction(cst, Δu, Δuᵖ)
    # Compute scalar tractions
    τ_tr = computeScalarTraction(cst, t_tr)
    # Pre return map update
    preReturnMap(cst, u_old, Δu)

    # Check yield conditions
    y = yieldFunction(cst, σ, τ_tr, 0.0)
    if (y <= 0.0)
        # Elastic update
        return (t_tr - t_old, computeTractionDerivative(cst, σ, t_tr, Δuᵖ))
    else
        # Plastic update
        Δp = returnMap(cst, σ, τ_tr)
        # Rebuild plastic DD vector from scalar plastic DD
        Δuᵖ = reformPlasticDD(cst, Δp, t_tr)
        # Post return map update
        postReturnMap(cst, Δp)
        # Jacobian
        return (computeTraction(cst, Δu, Δuᵖ), computeTractionDerivative(cst, t_tr, Δu, Δuᵖ))
    end
end

function returnMap(cst::AbstractFriction, σ::T, τ_tr::T)::T where {T<:Real}
    # Initialized scalar plastic DD
    Δp = 0.0

    # Initial residual
    res_ini = frictionResidual(cst, σ, τ_tr, Δp)

    res = res_ini
    jac = frictionJacobian(cst, σ, τ_tr, Δp)

    # Newton loop
    for iter in 1:cst.max_iter
        Δp -= res / jac

        res = frictionResidual(cst, σ, τ_tr, Δp)
        jac = frictionJacobian(cst, σ, τ_tr, Δp)

        # Convergence check
        if ((abs(res) <= cst.abs_tol) || (abs(res / res_ini) <= cst.rel_tol))
            return Δp
        end
    end
    throw(ErrorException("Plastic update failed after $(iter) iterations!"))
end

# Default traction calculations - 2D
function computeTraction(cst::AbstractFriction, Δδ::T, Δδᵖ::T)::T where {T<:Real}
    return cst.k * (Δδ - Δδᵖ)
end

function computeTractionDerivative(cst::AbstractFriction, σ::T, τ_tr::T, Δδᵖ::T)::T where {T<:Real}
    dΔδᵖ = computePlasticDDDerivative(cst, σ, τ_tr, Δδᵖ)
    return cst.k * (1.0 - dΔδᵖ)
end

# Default traction calculations - 3D
function computeTraction(cst::AbstractFriction, Δδ::SVector{2,T}, Δδᵖ::SVector{2,T})::SVector{2, T} where {T<:Real}
    return SVector(cst.k * (Δδ[1] - Δδᵖ[1]), cst.k * (Δδ[2] - Δδᵖ[2]))
end

function computeTractionDerivative(cst::AbstractFriction, σ::T, t_tr::SVector{2,T}, Δδᵖ::SVector{2,T})::SMatrix{2,2,T} where {T<:Real}
    dΔδᵖ = computePlasticDDDerivative(cst, σ, t_tr, computeScalarPlasticDD(cst, Δδᵖ))
    return SMatrix{2}(cst.k * (1.0 - dΔδᵖ[1, 1]), -cst.k * dΔδᵖ[1, 2], -cst.k * dΔδᵖ[2, 1], cst.k * (1.0 - dΔδᵖ[2, 2]))
end

# Default scalar traction calculations - 3D
function computeScalarTraction(cst::AbstractFriction, t_tr::SVector{2,T})::T where {T<:Real}
    return norm(t_tr)
end

# Default scalar plastic slip increment calculations - 3D
function computeScalarPlasticDD(cst::AbstractFriction, Δuᵖ::SVector{2,T})::T where {T<:Real}
    return norm(Δuᵖ)
end

# Default scalar to vector plastic DD
function reformPlasticDD(cst::AbstractFriction, Δp::T, t_tr::SVector{2,T})::SVector{2,T} where {T<:Real}
    τ_tr = computeScalarTraction(cst, t_tr)
    return SVector{2}(Δp * t_tr[1] / τ_tr, Δp * t_tr[2] / τ_tr)
end


# Default plastic flow direction - 3D
function plasticFlowDirection(cst::AbstractFriction, t_tr::SVector{2,T})::SVector{2,T} where {T<:Real}
    τ_tr = computeScalarTraction(cst, t_tr)
    return SVector{2}(t_tr[1] / τ_tr, t_tr[2] / τ_tr)
end

function plasticFlowDirectionDerivative(cst::AbstractFriction, t_tr::SVector{2,T})::SMatrix{2,2,T} where {T<:Real}
    τ_tr = computeScalarTraction(cst, t_tr)
    return SMatrix{2}(cst.k / τ_tr * (1.0 - (t_tr[1] / τ_tr)^2), -cst.k / τ_tr * t_tr[1] * t_tr[2] / τ_tr^2, 
        -cst.k / τ_tr * t_tr[1] * t_tr[2] / τ_tr^2, cst.k / τ_tr * (1.0 - (t_tr[2] / τ_tr)^2))
end

function plasticMultiplierDerivative(cst::AbstractFriction, σ::T, τ_tr::T, Δδᵖ::T)::T where {T<:Real}
    df_dτ = yieldFunctionStressDerivative(cst, σ, τ_tr, Δδᵖ)
    df_dΔp = yieldFunctionDerivative(cst, σ, τ_tr, Δδᵖ)
    return -cst.k * df_dτ / df_dΔp
end

function plasticMultiplierDerivative(cst::AbstractFriction, σ::T, t_tr::SVector{2,T}, Δp::T)::SVector{2,T} where {T<:Real}
    τ_tr = computeScalarTraction(cst, t_tr)
    df_dt = yieldFunctionStressDerivative(cst, σ, τ_tr, Δp)
    df_dΔp = yieldFunctionDerivative(cst, σ, τ_tr, Δp)
    return SVector{2}(-cst.k * df_dt * t_tr[1] / τ_tr / df_dΔp, -cst.k * df_dt * t_tr[2] / τ_tr / df_dΔp)
end

function computePlasticDDDerivative(cst::AbstractFriction, σ::T, τ_tr::T, Δδᵖ::T)::T where {T<:Real}
    if (Δδᵖ > 0)
        return plasticMultiplierDerivative(cst, σ, τ_tr, Δδᵖ)
    else
        return 0.0
    end
end

function computePlasticDDDerivative(cst::AbstractFriction, σ::T, t_tr::SVector{2,T}, Δp::T)::SMatrix{2,2,T} where {T<:Real}
    r = plasticFlowDirection(cst, t_tr)
    dΔp_dΔu = plasticMultiplierDerivative(cst, σ, t_tr, Δp)
    dr_dΔu = plasticFlowDirectionDerivative(cst, t_tr)
    if (Δp > 0.0)
        return SMatrix{2}(dΔp_dΔu[1] * r[1] + Δp * dr_dΔu[1,1], dΔp_dΔu[1] * r[2] + Δp * dr_dΔu[1,2],
            dΔp_dΔu[2] * r[1] + Δp * dr_dΔu[2,1], dΔp_dΔu[2] * r[2] + Δp * dr_dΔu[2,2])
    else
        return SMatrix{2}(0.0, 0.0, 0.0, 0.0)
    end
end

# Default residual and jacobian (plasticity)
function frictionResidual(cst::AbstractFriction, σ::T, τ_tr::T, Δp::T)::T where {T<:Real}
    return yieldFunction(cst, σ, τ_tr, Δp)
end

function frictionJacobian(cst::AbstractFriction, σ::T, τ_tr::T, Δp::T)::T where {T<:Real}
    return yieldFunctionDerivative(cst, σ, τ_tr, Δp)
end

# To replace in children classes
function yieldFunction(cst::AbstractFriction, σ::T, τ_tr::T, Δp::T) where {T<:Real}
    throw(MethodError(yieldFunction, cst))
end

function yieldFunctionDerivative(cst::AbstractFriction, σ::T, τ_tr::T, Δp::T) where {T<:Real}
    throw(MethodError(yieldFunctionDerivative, cst))
end

function yieldFunctionStressDerivative(cst::AbstractFriction, σ::T, τ_tr::T, Δp::T) where {T<:Real}
    throw(MethodError(yieldFunctionStressDerivative, cst))
end

function preReturnMap(cst::AbstractFriction, δ_old::T, Δδ::T) where {T<:Real}
    return nothing
end

function preReturnMap(cst::AbstractFriction, u_old::SVector{N,T}, Δu::SVector{N,T}) where {N,T<:Real}
    return nothing
end

function postReturnMap(cst::AbstractFriction, Δp::T) where {T<:Real}
    return nothing
end

# Constant yield model
struct ConstantYield{T<:Real} <: AbstractFriction
    " Shear stress yield"
    τ₀::T

    " Shear stiffness"
    k::T

    " Absolute tolerance"
    abs_tol::T

    " Relative tolerance"
    rel_tol::T

    " Maximum iterations"
    max_iter::Int

    " Constructor"
    function ConstantYield(τ₀::T, k::T; abs_tol::T=1.0e-12, rel_tol::T=1.0e-10, max_iter::Int=20) where {T<:Real}
        return new{T}(τ₀, k, abs_tol, rel_tol, max_iter)
    end
end

function yieldFunction(cst::ConstantYield{T}, σ::T, τ_tr::T, Δp::T)::T where {T<:Real}
    return (τ_tr - cst.k * Δp) - cst.τ₀
end

function yieldFunctionDerivative(cst::ConstantYield{T}, σ::T, τ_tr::T, Δp::T)::T where {T<:Real}
    return -cst.k
end

function yieldFunctionStressDerivative(cst::ConstantYield{T}, σ::T, τ_tr::T, Δp::T)::T where {T<:Real}
    return 1.0
end

# Constant friction model
struct ConstantFriction{T<:Real} <: AbstractFriction
    " Friction coefficient"
    f::T

    " Shear stiffness"
    k::T

    " Absolute tolerance"
    abs_tol::T

    " Relative tolerance"
    rel_tol::T

    " Maximum iterations"
    max_iter::Int

    " Constructor"
    function ConstantFriction(f::T, k::T; abs_tol::T=1.0e-12, rel_tol::T=1.0e-10, max_iter::Int=20) where {T<:Real}
        return new{T}(f, k, abs_tol, rel_tol, max_iter)
    end
end

function yieldFunction(cst::ConstantFriction{T}, σ::T, τ_tr::T, Δp::T)::T where {T<:Real}
    return (τ_tr - cst.k * Δp) - cst.f * σ
end

function yieldFunctionDerivative(cst::ConstantFriction{T}, σ::T, τ_tr::T, Δp::T)::T where {T<:Real}
    return -cst.k
end

function yieldFunctionStressDerivative(cst::ConstantFriction{T}, σ::T, τ_tr::T, Δp::T)::T where {T<:Real}
    return 1.0
end

function reformPlasticDD(cst::ConstantFriction{T}, Δp::T, t_tr::SVector{2,T})::SVector{2,T} where {T<:Real}
    return Δp * plasticFlowDirection(cst, t_tr)
end

# Slip-weakening friction model
mutable struct SlipWeakeningFriction{T<:Real} <: AbstractFriction
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

    " Shear stiffness"
    k::T

    " Absolute tolerance"
    abs_tol::T

    " Relative tolerance"
    rel_tol::T

    " Maximum iterations"
    max_iter::Int

    " Constructor"
    function SlipWeakeningFriction(fₚ::T, fᵣ::T, δᵣ::T, k::T; abs_tol::T=1.0e-12, rel_tol::T=1.0e-10, max_iter::Int=20) where {T<:Real}
        return new{T}(0.0, fₚ, fᵣ, δᵣ, (fₚ - fᵣ) / δᵣ, k, abs_tol, rel_tol, max_iter)
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

function yieldFunction(cst::SlipWeakeningFriction{T}, σ::T, τ_tr::T, Δp::T)::T where {T<:Real}
    f = friction(cst, Δp)
    return (τ_tr - cst.k * Δp) - f * σ
end

function yieldFunctionDerivative(cst::SlipWeakeningFriction{T}, σ::T, τ_tr::T, Δp::T)::T where {T<:Real}
    f = friction(cst, Δp)
    df = frictionSlipDerivative(cst, Δp)
    return -cst.k - df * σ
end

function yieldFunctionStressDerivative(cst::SlipWeakeningFriction{T}, σ_tr::T, τ_tr::T, Δp::T)::T where {T<:Real}
    return 1.0
end

function reformPlasticDD(cst::SlipWeakeningFriction{T}, Δp::T, t_tr::SVector{2,T})::SVector{2,T} where {T<:Real}
    return Δp * plasticFlowDirection(cst, t_tr)
end

function preReturnMap(cst::SlipWeakeningFriction{T}, δ_old::T, Δδ::T) where {T<:Real}
    # Compute scalar old DD
    cst.δ_old = δ_old
    return nothing
end

function preReturnMap(cst::SlipWeakeningFriction{T}, u_old::SVector{N,T}, Δu::SVector{N,T}) where {N,T<:Real}
    # Compute scalar old DD
    cst.δ_old = computeScalarPlasticDD(cst, u_old)
    return nothing
end

function postReturnMap(cst::SlipWeakeningFriction{T}, Δp::T) where {T<:Real}
    return nothing
end