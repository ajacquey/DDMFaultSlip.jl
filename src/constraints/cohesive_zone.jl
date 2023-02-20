abstract type AbstractCohesiveZone{T<:Real} end

# For NormalDDProblem
function applyCohesiveZoneConstraints(cst::AbstractCohesiveZone{T}, Δu::T, X::SVector{N,T}, u_old::T, σ_old::T) where {N,T<:Real}
    # Plastic DD increment
    Δp = 0.0
    # Compute trial normal stress
    σ_tr = σ_old + computeTraction(cst, Δu, Δp)
    # Pre return map update
    preReturnMap(cst, u_old, Δu)

    # Check yield conditions
    y = yieldFunction(cst, σ_tr, 0.0, X)
    if (y <= 0.0)
        # Elastic update
        return (σ_tr - σ_old, computeTractionDerivative(cst, σ_tr, Δu, Δp, X))
    else
        # Plastic update
        Δp = returnMap(cst, σ_tr, X)
        # Post return map update
        postReturnMap(cst, Δp)
        # Jacobian
        return (computeTraction(cst, Δu, Δp), computeTractionDerivative(cst, σ_tr, Δu, Δp, X))
    end
end

# For ShearDDProblem3D
function applyCohesiveZoneConstraints(cst::AbstractCohesiveZone{T}, Δu::SVector{2,T}, X::SVector{3,T}, u_old::SVector{2,T}, t_old::SVector{2,T}) where {T<:Real}
    # Plastic DD increment
    Δuᵖ = @SVector zeros(T, 2)
    # Compute trial shear tractions
    t_tr = t_old + computeTraction(cst, Δu, Δuᵖ)
    # Compute trial shear stress
    τ_tr = computeScalarTraction(cst, t_tr)
    # Pre return map update
    preReturnMap(cst, u_old, Δu)

    # Check yield conditions
    y = yieldFunction(cst, τ_tr, 0.0, X)
    if (y <= 0.0)
        # Elastic update
        return (t_tr - t_old, computeTractionDerivative(cst, t_tr, Δu, Δuᵖ, X))
    else
        # Plastic update
        Δp = returnMap(cst, τ_tr, X)
        # Rebuild plastic DD vector from scalar plastic DD
        Δuᵖ = reformPlasticDD(cst, Δp, t_tr)
        # Post return map update
        postReturnMap(cst, Δp)
        # Jacobian
        return (computeTraction(cst, Δu, Δuᵖ), computeTractionDerivative(cst, t_tr, Δu, Δuᵖ, X))
    end
end

function returnMap(cst::AbstractCohesiveZone{T}, σ_tr::T, X::SVector{N,T})::T where {N,T<:Real}
    # Initialized scalar plastic DD
    Δp = 0.0

    # Initial residual
    res_ini = cohesiveZoneResidual(cst, σ_tr, Δp, X)

    res = res_ini
    jac = cohesiveZoneJacobian(cst, σ_tr, Δp, X)

    # Newton loop
    for iter in 1:cst.max_iter
        Δp -= res / jac

        res = cohesiveZoneResidual(cst, σ_tr, Δp, X)
        jac = cohesiveZoneJacobian(cst, σ_tr, Δp, X)

        # Convergence check
        if ((abs(res) <= cst.abs_tol) || (abs(res / res_ini) <= cst.rel_tol))
            return Δp
        end
    end
    throw(ErrorException("Plastic update failed after $(iter) iterations!"))
end

# Default traction calculations - 2D
function computeTraction(cst::AbstractCohesiveZone{T}, Δu::T, Δp::T)::T where {T<:Real}
    return cst.k * (Δu - Δp)
end

function computeTractionDerivative(cst::AbstractCohesiveZone{T}, σ_tr::T, Δu::T, Δp::T, X::SVector{N,T})::T where {N,T<:Real}
    dΔp = computePlasticDDDerivative(cst, σ_tr, Δp, X)
    return cst.k * (1.0 - dΔp)
end

# Default traction calculations - 3D
function computeTraction(cst::AbstractCohesiveZone{T}, Δu::SVector{2,T}, Δuᵖ::SVector{2,T})::SVector{2,T} where {T<:Real}
    return cst.k * (Δu - Δuᵖ)
end

function computeTractionDerivative(cst::AbstractCohesiveZone{T}, t_tr::SVector{2,T}, Δu::SVector{2,T}, Δuᵖ::SVector{2,T}, X::SVector{3,T})::SMatrix{2,2,T} where {T<:Real}
    dΔuᵖ = computePlasticDDDerivative(cst, t_tr, computeScalarPlasticDD(cst, Δuᵖ), X)
    return SMatrix{2}(cst.k * (1.0 - dΔuᵖ[1, 1]), -cst.k * dΔuᵖ[2, 1], -cst.k * dΔuᵖ[1, 2], cst.k * (1.0 - dΔuᵖ[2, 2]))
end

# Default scalar traction calculations - 3D
function computeScalarTraction(cst::AbstractCohesiveZone{T}, t_tr::SVector{2,T})::T where {T<:Real}
    return norm(t_tr)
end

# Default scalar plastic slip increment calculations - 3D
function computeScalarPlasticDD(cst::AbstractCohesiveZone{T}, Δuᵖ::SVector{2,T})::T where {T<:Real}
    return norm(Δuᵖ)
end

# Default scalar to vector plastic DD - 3D
function reformPlasticDD(cst::AbstractCohesiveZone{T}, Δp::T, t_tr::SVector{2,T})::SVector{2,T} where {T<:Real}
    τ_tr = computeScalarTraction(cst, t_tr)
    return Δp * t_tr / τ_tr
end

# Default plastic flow direction - 3D
function plasticFlowDirection(cst::AbstractCohesiveZone{T}, t_tr::SVector{2,T})::SVector{2,T} where {T<:Real}
    τ_tr = computeScalarTraction(cst, t_tr)
    return t_tr / τ_tr
end

function plasticFlowDirectionDerivative(cst::AbstractCohesiveZone{T}, t_tr::SVector{2,T})::SMatrix{2,2,T} where {T<:Real}
    τ_tr = computeScalarTraction(cst, t_tr)
    return SMatrix{2}(cst.k / τ_tr * (1.0 - (t_tr[1] / τ_tr)^2), -cst.k / τ_tr * t_tr[1] * t_tr[2] / τ_tr^2, -cst.k / τ_tr * t_tr[1] * t_tr[2] / τ_tr^2, cst.k / τ_tr * (1.0 - (t_tr[2] / τ_tr)^2))
end

# Jacobian NormalDDProblem - 2D
function computePlasticDDDerivative(cst::AbstractCohesiveZone{T}, σ_tr::T, Δp::T, X::SVector{N,T})::T where {N,T<:Real}
    dΔp_dΔu = plasticMultiplierDerivative(cst, σ_tr, Δp, X)

    if Δp > 0.0
        return dΔp_dΔu
    else
        return 0.0
    end
end

function plasticMultiplierDerivative(cst::AbstractCohesiveZone{T}, σ_tr::T, Δp::T, X::SVector{N,T})::T where {N,T<:Real}
    df_dσ = yieldFunctionStressDerivative(cst, σ_tr, Δp, X)
    df_dΔp = yieldFunctionDerivative(cst, σ_tr, Δp, X)
    return -cst.k * df_dσ / df_dΔp
end

# Jacobian - 3D
function computePlasticDDDerivative(cst::AbstractCohesiveZone{T}, t_tr::SVector{2,T}, Δp::T, X::SVector{3,T})::SMatrix{2,2,T} where {T<:Real}
    r = plasticFlowDirection(cst, t_tr)
    dΔp_dΔu = plasticMultiplierDerivative(cst, t_tr, Δp, X)
    dr_dΔu = plasticFlowDirectionDerivative(cst, t_tr)
    if Δp > 0.0
        return SMatrix{2}(dΔp_dΔu[1] * r[1] + Δp * dr_dΔu[1,1], dΔp_dΔu[2] * r[1] + Δp * dr_dΔu[2,1],
            dΔp_dΔu[1] * r[2] + Δp * dr_dΔu[1,2], dΔp_dΔu[2] * r[2] + Δp * dr_dΔu[2,2])
    else
        return SMatrix{2}(0.0, 0.0, 0.0, 0.0)
    end
end

function plasticMultiplierDerivative(cst::AbstractCohesiveZone{T}, t_tr::SVector{2,T}, Δp::T, X::SVector{3,T})::SVector{2,T} where {T<:Real}
    τ_tr = computeScalarTraction(cst, t_tr)
    df_dt = yieldFunctionStressDerivative(cst, τ_tr, Δp, X)
    df_dΔp = yieldFunctionDerivative(cst, τ_tr, Δp, X)
    return SVector{2}(-cst.k * df_dt * t_tr[1] / τ_tr / df_dΔp, -cst.k * df_dt * t_tr[2] / τ_tr / df_dΔp)
end

# Default residual and jacobian (plasticity)
function cohesiveZoneResidual(cst::AbstractCohesiveZone{T}, σ_tr::T, Δp::T, X::SVector{N,T})::T where {N,T<:Real}
    return yieldFunction(cst, σ_tr, Δp, X)
end

function cohesiveZoneJacobian(cst::AbstractCohesiveZone{T}, σ_tr::T, Δp::T, X::SVector{N,T})::T where {N,T<:Real}
    return yieldFunctionDerivative(cst, σ_tr, Δp, X)
end

# To replace in children classes
function yieldFunction(cst::AbstractCohesiveZone{T}, σ_tr::T, Δp::T, X::SVector{N,T}) where {N,T<:Real}
    throw(MethodError(yieldFunction, cst))
end

function yieldFunctionDerivative(cst::AbstractCohesiveZone{T}, σ_tr::T, Δp::T, X::SVector{N,T}) where {N,T<:Real}
    throw(MethodError(yieldFunctionDerivative, cst))
end

function yieldFunctionStressDerivative(cst::AbstractCohesiveZone{T}, σ_tr::T, Δp::T, X::SVector{N,T}) where {N,T<:Real}
    throw(MethodError(yieldFunctionStressDerivative, cst))
end

function preReturnMap(cst::AbstractCohesiveZone{T}, u_old::T, Δu::T) where {T<:Real}
    return nothing
end

function preReturnMap(cst::AbstractCohesiveZone{T}, u_old::SVector{2,T}, Δu::SVector{2,T}) where {T<:Real}
    return nothing
end

function postReturnMap(cst::AbstractCohesiveZone{T}, Δp::T) where {T<:Real}
    return nothing
end

# Dugdale cohesive zone model
struct DugdaleCohesiveZone{T<:Real} <: AbstractCohesiveZone{T}
    " Peak stress"
    σₚ::T

    " Residual stress"
    σᵣ::T

    " Open radius of the crack"
    l::T

    " Stiffness"
    k::T

    " Absolute tolerance"
    abs_tol::T

    " Relative tolerance"
    rel_tol::T

    " Maximum iterations"
    max_iter::Int

    " Constructor"
    function DugdaleCohesiveZone(σₚ::T, σᵣ::T, l::T, k::T; abs_tol::T=1.0e-12, rel_tol::T=1.0e-10, max_iter::Int=20) where {T<:Real}
        return new{T}(σₚ, σᵣ, l, k, abs_tol, rel_tol, max_iter)
    end
end

function yieldFunction(cst::DugdaleCohesiveZone{T}, σ_tr::T, Δp::T, X::SVector{N,T})::T where {N,T<:Real}
    if norm(X) <= cst.l
        return (σ_tr - cst.k * Δp) - cst.σᵣ
    else
        return (σ_tr - cst.k * Δp) - cst.σₚ
    end
end

function yieldFunctionDerivative(cst::DugdaleCohesiveZone{T}, σ_tr::T, Δp::T, X::SVector{N,T})::T where {N,T<:Real}
    return -cst.k
end

function yieldFunctionStressDerivative(cst::DugdaleCohesiveZone{T}, σ_tr::T, Δp::T, X::SVector{N,T})::T where {N,T<:Real}
    return 1.0
end
