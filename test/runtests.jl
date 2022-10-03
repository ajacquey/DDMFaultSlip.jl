using DDMFaultSlip
using Test

@testset "DDMFaultSlip.jl" begin
    include("opening.jl")
    include("shearing.jl")
end
