module Axis

include("ffi.jl")
include("math/vector.jl")

#=
Core Dispatcher API - generic WebGPU compute dispatch
=#
export BINDING_STORAGE_READ, BINDING_STORAGE_READ_WRITE, BINDING_UNIFORM
export wgpu_init!
export wgpu_create_buffer!, wgpu_write_buffer!, wgpu_read_buffer!, wgpu_destroy_buffer!
export wgpu_create_compute_pipeline!, wgpu_bind_buffers!, wgpu_dispatch!, wgpu_destroy_pipeline!
export axis_rs_library_path, axis_rs_available
export normalize2

#= 
Ocean - built-in physics module ( reference implementation )
Shows how to build a physics module on top of the Axis dispatcher.
Usage : using Axis.Ocean  OR  import Axis.Ocean as AXOcean
=#
module Ocean
    import ..Axis: BINDING_STORAGE_READ, BINDING_STORAGE_READ_WRITE, BINDING_UNIFORM
    import ..Axis: wgpu_init!, wgpu_create_buffer!, wgpu_write_buffer!
    import ..Axis: wgpu_read_buffer!, wgpu_destroy_buffer!
    import ..Axis: wgpu_create_compute_pipeline!, wgpu_bind_buffers!
    import ..Axis: wgpu_dispatch!, wgpu_destroy_pipeline!
    import ..Axis: normalize2

    #= 
    Ocean simulation interface - all concrete simulators subtype this.
    Enables multiple dispatch and type-stable collections of mixed simulators.
    =#
    abstract type AbstractOceanSim end

    include("ocean/phillips_ocean.jl")

    export AbstractOceanSim
    export PhillipsSim
    export create_phillips_sim, destroy!
    export init!, compute_wave!
    export phillips_spectrum, build_components!, precompute_phase!, upload_buffers!
end

export Ocean

function __init__()
    _init_axis_rs!()
    return nothing
end

end # module Axis
