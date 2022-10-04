const Point2D{T<:Real} = SVector{2, T}
const Point3D{T<:Real} = SVector{3, T}

abstract type DDElem{T<:Real} end

"""
One-dimensional element: Edge
"""
struct DDEdgeElem{T<:Real} <: DDElem{T}
    " List of nodes"
    nodes::SVector{2, Point2D{T}}
    " Centroid coordinates"
    X::Point2D{T}
end

"""
Two-dimensional element: Triangle
"""
struct DDTriangleElem{T<:Real} <: DDElem{T}
    " List of nodes"
    nodes::SVector{3, Point3D{T}}
    " Centroid coordinates"
    X::Point3D{T}
end

abstract type DDMesh{T<:Real} end

"""
One-dimensional DDMesh
"""
struct DDMesh1D{T<:Real} <: DDMesh{T}
    " List of nodes"
    nodes::Vector{Point2D{T}}
    " List of elems"
    elems::Vector{DDEdgeElem{T}}
    " Elements connection"
    elem2nodes::Vector{SVector{2, Int}}
    " Constructor"
    function DDMesh1D(start_point::SVector{2, T}, end_point::SVector{2, T}, n::Int) where {T<:Real}
        # Directional vector
        u = (end_point .- start_point) / n
        # Nodes list
        nodes = [start_point .+ (i - 1) * u for i in 1:n+1]
        # Elems list
        elems = [DDEdgeElem(SVector(nodes[i], nodes[i+1]), (nodes[i] + nodes[i+1]) / 2.0) for i in 1:n]
        # Elements connection
        elem2nodes = [[i, i+1] for i in 1:n]

        return new{T}(nodes, elems, elem2nodes)
    end
end

"""
Custom `show` function for `DDMesh1D{T}` that prints some information.
"""
function Base.show(io::IO, mesh::DDMesh1D{T} where {T<:Real})
    @printf("Mesh information:\n")
    @printf("  Mesh dimension: 1\n")
    @printf("  Nodes:          %i\n", length(mesh.nodes))
    @printf("  Elements:       %i\n\n", length(mesh.elems))
end

"""
Two-dimensional DDMesh
"""
struct DDMesh2D{T<:Real} <: DDMesh{T}
    " List of nodes"
    nodes::Vector{Point3D{T}}
    " List of elems"
    elems::Vector{DDTriangleElem{T}}
    " Elements connection"
    elem2nodes::Vector{SVector{3, Int}}
    " Constructor"
    function DDMesh2D(T::DataType, file::String; log::Bool = false)
        # Check if file exists
        @assert isfile(file)
        # Create containers
        nodes = Vector{Point3D{T}}(undef, 0)
        nodes_id = Vector{Int}(undef, 0)
        elems = Vector{DDTriangleElem}(undef, 0)
        elem2nodes = Vector{Vector{Int}}(undef, 0)
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
                    resize!(elem2nodes, n_elems)
                    for i in 1:length(elem2nodes)
                        elem2nodes[i] = Vector{Int}(undef, 0)
                    end
                    line = strip(readline(f))
                    while ~startswith(line, string("\$End", keyword))
                        # Save node connectivity
                        elem_co = parse.(Int, split(line, " "))
                        resize!(elem2nodes[elem_co[1]], length(elem_co) - 1)
                        elem2nodes[elem_co[1]] = elem_co[2:end]
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
        # Build list of elements
        n_elems = length(elem2nodes)
        resize!(elems, n_elems)
        for k in 1:n_elems
            elems[k] = DDTriangleElem(SVector(nodes[elem2nodes[k][1]], nodes[elem2nodes[k][2]], nodes[elem2nodes[k][3]]), (nodes[elem2nodes[k][1]] + nodes[elem2nodes[k][2]] + nodes[elem2nodes[k][3]]) / 3.0)
        end
        if log
            @printf("\n-> Done reading %s with %i nodes and %i elements.\n\n", file, length(nodes), length(elems))
        end
        return new{T}(nodes, elems, elem2nodes)
    end
end

"""
Custom `show` function for `DDMesh2D{T}` that prints some information.
"""
function Base.show(io::IO, mesh::DDMesh2D{T} where {T<:Real})
    @printf("Mesh information:\n")
    @printf("  Mesh dimension: 2\n")
    @printf("  Nodes:          %i\n", length(mesh.nodes))
    @printf("  Elements:       %i\n\n", length(mesh.elems))
end