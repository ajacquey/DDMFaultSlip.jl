struct FunctionConstraint<: AbstractConstraint
    " Residual function"
    func_res::Function
    
    " Jacobian function"
    func_jac::Function

    " Constructors"
    function FunctionConstraint(func_res::Function)
        return new(func_res, (x, y) -> 0.0)
    end
    function FunctionConstraint(func_res::Function, func_jac::Function)
        return new(func_res, func_jac)
    end
end

function computeConstraints(cst::FunctionConstraint, time::T, X::SVector{2, T})::Tuple{T, T} where {T<:Real}
    return (cst.func_res(X, time), cst.func_jac(X, time))
end

function computeConstraints(cst::FunctionConstraint, time::T, X::SVector{3, T})::Tuple{T, T} where {T<:Real}
    return (cst.func_res(X, time), cst.func_jac(X, time))
end