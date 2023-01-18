using DDMFaultSlip
using Test

@testset "DDMFaultSlip.jl" begin
    include("opening.jl")
    include("generic.jl")
    include("shearing.jl")
    include("injection.jl")
    include("coupled.jl")
    include("fluid_coupling.jl")
    include("constant_friction.jl")
    include("slip_weakening_friction.jl")
    include("outputs.jl")
end
