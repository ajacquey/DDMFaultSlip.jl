abstract type Collocation{T<:Real} end

"""
Structure for collocation points
"""
struct Collocation1D{T<:Real} <: Collocation{T}
    # Coordinates of the collocation points
    cpoints::Vector{Point2D{T}}
    # Element half-lengths
    a::Vector{T}
    # Oder
    order::Int
    # Constructor
    function Collocation1D(mesh::Mesh1D{T}; order::Int=0) where {T<:Real}
        # Collocation points coordinates
        cpoints = collocation_points(mesh; order)
        # Element half-lengths
        a = element_half_length(mesh, order)

        return new{T}(cpoints, a, order)
    end
end

function collocation_points(mesh::Mesh{T}; order::Int=0) where {T<:Real}
    # Declare containers
    cps = Vector{eltype(mesh.nodes)}(undef, 0)
    # Piecewise Constant (PWC)
    if order == 0
        # Number of collocation points
        resize!(cps, length(mesh.elems))

        # Loop over elements
        Threads.@threads for i in 1:length(mesh.elems)
            cps[i] = zero(eltype(mesh.nodes))
            n = length(mesh.elems[i])
            for idx in mesh.elems[i]
                cps[i] = cps[i] .+ mesh.nodes[idx]
            end 
            cps[i] = cps[i] / n
        end
    else
        ErrorException("Order not supported!")
    end
    
    return cps
end

function element_half_length(mesh::Mesh1D{T}, order::Int=0) where {T<:Real}
    # Declare containers
    a = Vector{T}(undef, 0)
    # Piecewise Constant (PWC)
    if order == 0
        # Number of collocation points
        resize!(a, length(mesh.elems))

        # Loop over elements
        Threads.@threads for i in 1:length(mesh.elems)
            # Specific to 1D elements
            a[i] = norm(mesh.nodes[2] .- mesh.nodes[1])
        end
    else
        ErrorException("Order not supported!")
    end

    return a
end

"""
Custom `show` function for `Collocation1D{T}` that prints some information.
"""
function Base.show(io::IO, cps::Collocation1D{T} where {T<:Real})
    println("Collocation points information:")
    println("   -> dimension: 1")
    println("   -> order: $(cps.order)")
    println("   -> n_cps: $(length(cps.cpoints))")
end

abstract type ElasticKernelMatrix{T<:Real} <: AbstractMatrix{T} end

"""
Piecewise constant (PWC) one-dimensional elastic kernel matrix
"""
struct PWC1DElasticMatrix{T<:Real} <: ElasticKernelMatrix{T}
    cpi::Collocation1D{T}
    cpj::Collocation1D{T}
end

function Base.getindex(K::PWC1DElasticMatrix, i::Int, j::Int)
    return 2.0 * norm(K.cpj.a[j]) / (π * (norm(K.cpi.cpoints[i] .- K.cpj.cpoints[j])^2 - norm(K.cpj.a)^2))
end

function Base.size(K::PWC1DElasticMatrix)
    return length(K.cpi.cpoints), length(K.cpj.cpoints)
end