abstract type AbstractFluidCoupling{T<:Real} end

# mutable struct FunctionPressure{T<:Real} <: AbstractFluidCoupling{T}
#     " Pressure variable"
#     p::AuxVariable{T}

#     " Function to udate pressure"
#     fct::Function

#     " Constructor"
#     function FunctionPressure(problem::AbstractDDProblem{T}, fct::Function; fct_ic::Function = default_ic) where {T<:Real}
#         # Pressure auxiliary variable
#         p = AuxVariable(T, :p, length(problem.mesh.elems))

#         # Initial pressure value
#         p.value = fct_ic.([problem.mesh.elems[i].X for i in 1:length(p.value)], 0.0)
#         p.value_old = copy(p.value)

#         return new{T}(p, fct)
#     end
# end

# function default_ic(X::Point2D{T}) where {T<:Real}
#     return 0.0
# end

# function default_ic(X::Point3D{T}) where {T<:Real}
#     return 0.0
# end