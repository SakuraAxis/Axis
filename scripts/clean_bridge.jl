#= CMD : julia --project=. scripts/clean_bridge.jl =#

import Axis as AX

function main()
    axis_rs_path = abspath(joinpath(@__DIR__, "..", "axis_rs"))
    @info "Starting to clean bridge at: $axis_rs_path"
    try
        AX.bridge_down(axis_rs_path)
        @info "Bridge cleanup completed successfully!"
    catch e
        @error "Failed to clean bridge:" exception=(e, catch_backtrace())
    end
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
