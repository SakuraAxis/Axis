module Axis

#=
Axis - a lightweight Julia -> Rust FFI bridge + wgpu compute dispatcher.

Two distinct responsibilities :
  1. WGPU Dispatcher  : thin Julia wrappers around the axis_rs C-ABI, providing
                        buffer/pipeline/dispatch operations with zero overhead.
  2. Rust Code Generator : @rust_fn / @call_rust_fn / @rust_code macros + generate_bridge(),
                            letting callers embed Rust functions directly in their
                            Julia source files and emit professional Rust modules.

Usage example ( in Achernar's PhillipsOceanAX.jl ) :

    import Axis as AX

    @AX.rust_code \"\"\"
    // any Rust helpers, structs, use statements ...
    \"\"\"

    @AX.rust_fn function pack_components!(kx::Ptr{Float32}, t::Float32, dest::Ptr{Float32}, n::Int32)::Cvoid
        \"\"\"
        let n = n as usize;
        let kx  = unsafe { std::slice::from_raw_parts(kx, n) };
        let out = unsafe { std::slice::from_raw_parts_mut(dest, n * 4) };
        for i in 0..n { out[i * 4] = kx[i]; }
        \"\"\"
    end

    @AX.call_rust_fn pack_components!(pointer(kx), 0f0, pointer(out), Int32(length(kx)))

    # After all modules are loaded :
    AX.generate_bridge("path/to/axis_rs")
    # Then : cargo build --release  in axis_rs/
=#

include("types.jl")   # Julia -> Rust type mapping & name helpers
include("codegen.jl") # @rust_fn, @call_rust_fn, @rust_code, generate_bridge
include("ffi.jl")     # wgpu_init!, wgpu_create_buffer!, wgpu_dispatch!, …

#=
WGPU Dispatcher public API
=#
export BINDING_STORAGE_READ, BINDING_STORAGE_READ_WRITE, BINDING_UNIFORM
export wgpu_init!
export wgpu_create_buffer!, wgpu_write_buffer!, wgpu_read_buffer!, wgpu_destroy_buffer!
export wgpu_create_compute_pipeline!, wgpu_bind_buffers!, wgpu_dispatch!, wgpu_destroy_pipeline!
export axis_rs_library_path, axis_rs_available, bridge_up, bridge_down

#=
Math utilities
=#

#=
Rust Code Generator public API
  @rust_fn, @call_rust_fn, @rust_code are accessible via AX-qualified macros.
  generate_bridge is a regular function.
=#
export generate_bridge

function __init__()
    _init_axis_rs!()
    return nothing
end

end # module Axis
