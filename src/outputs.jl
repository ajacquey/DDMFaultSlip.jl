abstract type AbstractOutput{T<:Real} end

function initializeOutputs!(outputs::Vector{<:AbstractOutput{T}}, problem::AbstractDDProblem{T}, output_initial::Bool) where {T<:Real}
    for out in outputs
        initialize!(out, problem, output_initial)
    end

    return nothing
end

function executeOutputs!(outputs::Vector{<:AbstractOutput{T}}, problem::AbstractDDProblem{T}, exec::AbstractExecutioner{T}, output_initial::Bool) where {T<:Real}
    for out in outputs
        execute!(out, problem, exec, output_initial)
    end

    return nothing
end

struct VTKDomainOutput{T<:Real} <: AbstractOutput{T}
    " Base file name"
    filename_base::String

    " List of points"
    vtk_points::Matrix{T}

    " List of cells"
    vtk_cells::Vector{MeshCell{VTKCellType,Vector{Int}}}

    " Constructor"
    function VTKDomainOutput(mesh::DDMesh2D{T}, filename::String) where {T<:Real}
        # Create VTK grid
        n_points = length(mesh.nodes)
        n_cells = length(mesh.elems)

        # Create containers
        points = Matrix{T}(undef, (3, n_points))
        cells = Vector{MeshCell{VTKCellType,Vector{Int}}}(undef, n_cells)

        # List of nodes
        Threads.@threads for idx in eachindex(mesh.nodes)
            @inbounds points[:, idx] .= mesh.nodes[idx]
        end

        # List of elems
        Threads.@threads for idx in eachindex(mesh.elems)
            @inbounds cells[idx] = MeshCell(VTKCellTypes.VTK_TRIANGLE, Array(mesh.elem2nodes[idx]))
        end

        # Check filename and if sub-folder exists
        if (contains(filename, '/'))
            output_dir = string()
            k_end = findlast(isequal('/'), filename)
            if (startswith(filename, '/')) # absolute path
                output_dir = filename[1:k_end]
            else # relative path
                output_dir = string(dirname(Base.source_path()), "/", filename[1:k_end])
            end
            # Create folder if it doesn't exist
            if (!isdir(output_dir))
                mkpath(output_dir)
            end
            filename_base = string(output_dir, filename[k_end+1:end])
        else
            filename_base = string(dirname(Base.source_path()), "/", filename)
        end

        return new{T}(filename_base, points, cells)
    end
end

function createVTK(out::VTKDomainOutput{T}, problem::AbstractDDProblem{T}, time::T, dt::T, time_step::Int) where {T<:Real}
    vtk = vtk_grid(string(out.filename_base, "_", string(time_step)), out.vtk_points, out.vtk_cells; append=true, ascii=false, compress=true)
    if hasNormalDD(problem)
        vtk["epsilon", VTKCellData()] = problem.ϵ.value
        vtk["sigma", VTKCellData()] = problem.σ.value
        if (dt != 0.0)
            vtk["epsilon_dot", VTKCellData()] = (problem.ϵ.value - problem.ϵ.value_old) / dt
        else
            vtk["epsilon_dot", VTKCellData()] = zeros(T, length(problem.ϵ.value))
        end
    end
    if hasShearDD3D(problem)
        vtk["delta", VTKCellData()] = cat(problem.δ_x.value, problem.δ_y.value; dims=2)'
        vtk["tau", VTKCellData()] = cat(problem.τ_x.value, problem.τ_y.value; dims=2)'
        if (dt != 0.0)
            vtk["delta_dot", VTKCellData()] = cat((problem.δ_x.value - problem.δ_x.value_old) / dt, (problem.δ_y.value - problem.δ_y.value_old) / dt; dims=2)'
        else
            vtk["delta_dot", VTKCellData()] = zeros(T, 2, length(problem.ϵ.value))
        end
    end
    if hasFluidCoupling(problem)
        vtk["p", VTKCellData()] = problem.fluid_coupling[1].p
    end
    vtk["element_id", VTKCellData()] = collect(1:length(out.vtk_cells))
    vtk["node_id", VTKPointData()] = collect(1:size(out.vtk_points, 2))
    vtk["time", VTKFieldData()] = time

    return vtk
end

function initialize!(out::VTKDomainOutput{T}, problem::AbstractDDProblem{T}, output_initial::Bool) where {T<:Real}
    if output_initial
        pvd = paraview_collection(out.filename_base)
        vtk = createVTK(out, problem, 0.0, 0.0, 0)
        pvd[0.0] = vtk
        vtk_save(pvd)
    end

    return nothing
end

function execute!(out::VTKDomainOutput{T}, problem::AbstractDDProblem{T}, exec::AbstractExecutioner{T}, output_initial::Bool) where {T<:Real}
    pvd = paraview_collection(out.filename_base; append=output_initial)
    vtk = createVTK(out, problem, exec.time, exec.dt, exec.time_step)
    pvd[exec.time] = vtk
    vtk_save(pvd)

    return nothing
end

struct CSVDomainOutput{T<:Real} <: AbstractOutput{T}
    " Base file name"
    filename_base::String

    " List of cell centroids"
    csv_points::Matrix{T}

    " Constructor"
    function CSVDomainOutput(mesh::DDMesh1D{T}, filename::String) where {T<:Real}
        # Create list of centroids
        n_cells = length(mesh.elems)

        # Create containers
        points = Matrix{T}(undef, (n_cells, 2))

        # List of elems
        Threads.@threads for idx in eachindex(mesh.elems)
            @inbounds points[idx,1] = mesh.elems[idx].X[1]
            @inbounds points[idx,2] = mesh.elems[idx].X[2]
        end

        # Check filename and if sub-folder exists
        if (contains(filename, '/'))
            output_dir = string()
            k_end = findlast(isequal('/'), filename)
            if (startswith(filename, '/')) # absolute path
                output_dir = filename[1:k_end]
            else # relative path
                output_dir = string(dirname(Base.source_path()), "/", filename[1:k_end])
            end
            # Create folder if it doesn't exist
            if (!isdir(output_dir))
                mkpath(output_dir)
            end
            filename_base = string(output_dir, filename[k_end+1:end])
        else
            filename_base = string(dirname(Base.source_path()), "/", filename)
        end

        return new{T}(filename_base, points)
    end
end

function addDataToCSV!(data::Matrix{T}, header::String, problem::AbstractDDProblem{T}, dt::T) where {T<:Real}
    if hasNormalDD(problem)
        data = hcat(data, problem.ϵ.value)
        header = string(header, ",epsilon")
        data = hcat(data, problem.σ.value)
        header = string(header, ",sigma")
        if (dt != 0.0)
            data = hcat(data, (problem.ϵ.value - problem.ϵ.value_old) / dt)
        else
            data = hcat(data, zeros(T, length(problem.ϵ.value)))
        end
        header = string(header, ",epsilon_dot")
    end
    if hasShearDD2D(problem)
        data = hcat(data, problem.δ.value)
        header = string(header, ",delta")
        data = hcat(data, problem.τ.value)
        header = string(header, ",tau")
        if (dt != 0.0)
            data = hcat(data, (problem.δ.value - problem.δ.value_old) / dt)
        else
            data = hcat(data, zeros(T, length(problem.δ.value)))
        end
        header = string(header, ",delta_dot")
    end
    if hasFluidCoupling(problem)
        data = hcat(data, problem.fluid_coupling[1].p)
        header = string(header, ",p")
    end
    header = string(header, "\n")
    return data, header
end

function initialize!(out::CSVDomainOutput{T}, problem::AbstractDDProblem{T}, output_initial::Bool) where {T<:Real}
    if output_initial
        data = out.csv_points
        header = "x,y"
        data, header = addDataToCSV!(data, header, problem, 0.0)

        open(string(out.filename_base, "_0.csv"); write = true) do f    
            write(f, header) # write header
            writedlm(f, data, ',') # write data
        end
    end

    return nothing
end

function execute!(out::CSVDomainOutput{T}, problem::AbstractDDProblem{T}, exec::AbstractExecutioner{T}, output_initial::Bool) where {T<:Real}
    data = out.csv_points
    header = "x,y"
    data, header = addDataToCSV!(data, header, problem, exec.dt)

    open(string(out.filename_base, "_", exec.time_step, ".csv"); write = true) do f
        write(f, header) # write header
        writedlm(f, data, ',') # write data
    end

    return nothing
end