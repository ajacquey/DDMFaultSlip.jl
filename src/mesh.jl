const Point2D{T<:Real} = SVector{2, T}
const Point3D{T<:Real} = SVector{3, T}

abstract type Mesh{T<:Real} end

"""
One-dimensional Mesh
"""
struct Mesh1D{T<:Real} <: Mesh{T}
    # List of nodes
    nodes::Vector{Point2D{T}}
    # Elements connection
    elems::Vector{Vector{Int}}
    # Constructor
    function Mesh1D(start_point::SVector{2, T}, end_point::SVector{2, T}, n::Int) where {T<:Real}
        # Directional vector
        u = (end_point .- start_point) / n
        # Nodes list
        nodes = [start_point .+ (i - 1) * u for i in 1:n+1]
        # Elements connection
        elems = [[i, i+1] for i in 1:n]

        return new{T}(nodes, elems)
    end
end

"""
Custom `show` function for `Mesh1D{T}` that prints some information.
"""
function Base.show(io::IO, mesh::Mesh1D{T} where {T<:Real})
    println("Mesh information:")
    println("   -> dimension: 1")
    println("   -> n_elems: $(length(mesh.elems))")
    println("   -> n_nodes: $(length(mesh.nodes))")
end

"""
Two-dimensional Mesh
"""
struct Mesh2D{T<:Real} <: Mesh{T}
    # List of nodes
    nodes::Vector{Point3D{T}}
    # Elements connection
    elems::Vector{Vector{Int}}
    # Constructor
    function Mesh2D(T::DataType, file::String)
        # Check if file exists
        @assert isfile(file)
        # Create containers
        nodes = Vector{Point3D{T}}(undef, 0)
        nodes_id = Vector{Int}(undef, 0)
        elems = Vector{Vector{Int}}(undef, 0)
        # Open file
        f = open(file, "r")
        # Read file line by line
        while !eof(f)
            line = strip(readline(f))
            if startswith(line, "\$")
                # Keyword
                keyword = chop(line; head=1, tail=0)
                # MeshFormat
                if keyword == "MeshFormat"
                    # Check that mesh format is 4.1
                    line = strip(readline(f))
                    mesh_format = split(line, " ")
                    @assert mesh_format[1] == "4.1"
                    while ~startswith(line, string("\$End", keyword))
                        line = strip(readline(f))
                    end
                elseif keyword == "Nodes"
                    # Read total number of nodes
                    line = strip(readline(f))
                    node_info = split(line, " ")
                    n_nodes = parse(Int, node_info[2])
                    # Resize containers
                    resize!(nodes, n_nodes)
                    resize!(nodes_id, n_nodes)
                    idx_co = 1
                    idx_id = 1
                    line = strip(readline(f))
                    while ~startswith(line, string("\$End", keyword))
                        # Save nodes
                        if length(split(line, " ")) == 1
                            nodes_id[idx_id] = parse(Int, line)
                            idx_id += 1
                        elseif length(split(line, " ")) == 3
                            nodes[nodes_id[idx_co]] = Point3D(parse.(T, split(line, " ")))
                            idx_co += 1
                        end
                        line = strip(readline(f))
                    end
                elseif keyword == "Elements"
                    # Read total number of elements
                    line = strip(readline(f))
                    elem_info = split(line, " ")
                    n_elems = parse(Int, elem_info[2])
                    # Resize containers and initialize elem 2 node
                    resize!(elems, n_elems)
                    for i in 1:length(elems)
                        elems[i] = Vector{Int}(undef, 0)
                    end
                    line = strip(readline(f))
                    while ~startswith(line, string("\$End", keyword))
                        # Save node connectivity
                        elem_co = parse.(Int, split(line, " "))
                        resize!(elems[elem_co[1]], length(elem_co) - 1)
                        elems[elem_co[1]] = elem_co[2:end]
                        line = strip(readline(f))
                    end
                else
                    while ~startswith(line, string("\$End", keyword))
                        line = strip(readline(f))
                    end
                    continue
                end
            end
        end
        println()
        println("-> Done reading a mesh with ", length(nodes), " nodes and ", length(elems), " elements.")
        println()
        return new{T}(nodes, elems)
    end
end

"""
Custom `show` function for `Mesh2D{T}` that prints some information.
"""
function Base.show(io::IO, mesh::Mesh2D{T} where {T<:Real})
    println("Mesh information:")
    println("   -> dimension: 2")
    println("   -> n_elems: $(length(mesh.elems))")
    println("   -> n_nodes: $(length(mesh.nodes))")
end