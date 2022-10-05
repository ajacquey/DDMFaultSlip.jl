using DDMFaultSlip
using Test

@testset "DDMFaultSlip.jl" begin
    include("opening.jl")
    include("generic.jl")
    include("shearing.jl")
    include("injection.jl")
    include("coupled.jl")
end
