abstract type AbstractFriction{T<:Real} end

struct ConstantFriction{T} <: AbstractFriction{T}
    " Friction coefficient"
    f::T

    " Shear stiffness"
    kₛ::T

    " Normal stiffness"
    kₙ::T
end