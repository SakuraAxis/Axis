using Libdl

const _LIB_NAME = Sys.iswindows() ? "axis_rs.dll" :
                  Sys.isapple()   ? "libaxis_rs.dylib" :
                                    "libaxis_rs.so"
const _AXIS_RS_LIB    = Ref{String}("")
const _AXIS_RS_HANDLE = Ref{Ptr{Cvoid}}(C_NULL)
const _AXIS_RS_SYMBOLS = IdDict{Symbol, Ptr{Cvoid}}()

function _default_axis_rs_library_path()
    package_dir   = dirname(@__DIR__)
    workspace_dir = dirname(package_dir)
    return joinpath(workspace_dir, "axis_rs", "target", "release", _LIB_NAME)
end

function _resolve_axis_rs_library_path()
    env_path = get(ENV, "AXIS_RS_LIB", "")
    isempty(env_path) || return env_path
    return _default_axis_rs_library_path()
end

function _init_axis_rs!()
    empty!(_AXIS_RS_SYMBOLS)
    _AXIS_RS_HANDLE[] = C_NULL

    lib_path = _resolve_axis_rs_library_path()
    _AXIS_RS_LIB[] = lib_path

    if isfile(lib_path)
        _AXIS_RS_HANDLE[] = Libdl.dlopen(lib_path)
    end

    return nothing
end

"""
    axis_rs_library_path()

Return the Rust shared library path used by this package.
Set `ENV["AXIS_RS_LIB"]` before loading `Axis` to override it.
"""
axis_rs_library_path() = _AXIS_RS_LIB[]

"""
    axis_rs_available()

Return `true` when the Rust shared library exists at `axis_rs_library_path()`.
"""
axis_rs_available() = isfile(_AXIS_RS_LIB[]) && _AXIS_RS_HANDLE[] != C_NULL

function _check_axis_rs_available()
    axis_rs_available() && return nothing
    error(
        "Axis Rust library was not found at $(_AXIS_RS_LIB[]). " *
        "Build it with `cargo build --release` in the `axis_rs` directory, " *
        "or set `ENV[\"AXIS_RS_LIB\"]` before `using Axis`.",
    )
end

function _axis_rs_symbol(name::Symbol)
    _check_axis_rs_available()
    return get!(_AXIS_RS_SYMBOLS, name) do
        Libdl.dlsym(_AXIS_RS_HANDLE[], name)
    end
end

#= 
Binding type constants ( must match wgpu_core.rs )
=#
"""Buffer binding type: storage read-only."""
const BINDING_STORAGE_READ       = UInt32(0)
"""Buffer binding type: storage read-write (supports readback)."""
const BINDING_STORAGE_READ_WRITE = UInt32(1)
"""Buffer binding type: uniform."""
const BINDING_UNIFORM            = UInt32(2)

#= 
Error code helpers
=#
function _check_wgpu_status(status::Cint, context::String)
    status == 0  && return nothing
    status == -1  && error("$context: null pointer passed to Rust.")
    status == -2  && error("$context: Axis WGPU dispatcher not initialized. Call wgpu_init!() first.")
    status == -3  && error("$context: resource ID already exists.")
    status == -4  && error("$context: resource ID not found.")
    status == -5  && error("$context: invalid argument (e.g. size=0 or bad binding type).")
    status == -11 && error("$context: no compatible GPU adapter found.")
    status == -12 && error("$context: failed to create GPU device.")
    status == -13 && error("$context: WGSL shader compilation failed.")
    status == -14 && error("$context: GPU buffer mapping failed.")
    status == -15 && error("$context: GPU device polling failed.")
    status == -16 && error("$context: GPU readback failed.")
    status == -17 && error("$context: Rust panicked during operation.")
    error("$context: unknown Rust error code $status.")
end

#= 
Dispatcher lifecycle
=#

"""
    wgpu_init!()

Initialise the WGPU device. Idempotent — safe to call multiple times.
"""
function wgpu_init!()
    status = ccall(_axis_rs_symbol(:rust_wgpu_init), Cint, ())
    _check_wgpu_status(status, "wgpu_init!")
    return nothing
end

#= 
Buffer management
=#

"""
    wgpu_create_buffer!(id, size_bytes, binding_type)

Allocate a GPU buffer of `size_bytes` bytes with the given binding type.
`binding_type` must be one of `BINDING_STORAGE_READ`, `BINDING_STORAGE_READ_WRITE`,
or `BINDING_UNIFORM`.
"""
function wgpu_create_buffer!(id::Integer, size_bytes::Integer, binding_type::UInt32)
    status = ccall(
        _axis_rs_symbol(:rust_wgpu_create_buffer),
        Cint,
        (Csize_t, Csize_t, UInt32),
        Csize_t(id), Csize_t(size_bytes), binding_type,
    )
    _check_wgpu_status(status, "wgpu_create_buffer!")
    return nothing
end

"""
    wgpu_write_buffer!(id, data)

Write a Julia `Array` or `Vector` to GPU buffer `id`. Zero-copy pointer pass.
"""
function wgpu_write_buffer!(id::Integer, data::Union{Array, Vector})
    byte_len = sizeof(data)
    status = ccall(
        _axis_rs_symbol(:rust_wgpu_write_buffer),
        Cint,
        (Csize_t, Ptr{UInt8}, Csize_t),
        Csize_t(id), Ptr{UInt8}(pointer(data)), Csize_t(byte_len),
    )
    _check_wgpu_status(status, "wgpu_write_buffer!")
    return nothing
end

"""
    wgpu_read_buffer!(id, dest)

Read GPU buffer `id` results into a pre-allocated Julia `Array` or `Vector`.
Only valid for `BINDING_STORAGE_READ_WRITE` buffers.
"""
function wgpu_read_buffer!(id::Integer, dest::Union{Array, Vector})
    byte_len = sizeof(dest)
    status = ccall(
        _axis_rs_symbol(:rust_wgpu_read_buffer),
        Cint,
        (Csize_t, Ptr{UInt8}, Csize_t),
        Csize_t(id), Ptr{UInt8}(pointer(dest)), Csize_t(byte_len),
    )
    _check_wgpu_status(status, "wgpu_read_buffer!")
    return nothing
end

"""
    wgpu_destroy_buffer!(id)

Destroy GPU buffer `id` and free its VRAM.
"""
function wgpu_destroy_buffer!(id::Integer)
    status = ccall(
        _axis_rs_symbol(:rust_wgpu_destroy_buffer),
        Cint,
        (Csize_t,),
        Csize_t(id),
    )
    _check_wgpu_status(status, "wgpu_destroy_buffer!")
    return nothing
end

#= 
Pipeline management
=#

"""
    wgpu_create_compute_pipeline!(id, wgsl_source, entry_point, binding_flags)

Compile and register a WGSL compute shader as pipeline `id`.

- `wgsl_source`    - WGSL source string
- `entry_point`    - entry-point function name ( e.g. `"main"` )
- `binding_flags`  - `Vector{UInt32}` with one flag per `@binding` in the shader
                     ( `BINDING_STORAGE_READ`, `BINDING_STORAGE_READ_WRITE`, or `BINDING_UNIFORM` )
"""
function wgpu_create_compute_pipeline!(
    id::Integer,
    wgsl_source::String,
    entry_point::String,
    binding_flags::Vector{UInt32},
)
    status = ccall(
        _axis_rs_symbol(:rust_wgpu_create_compute_pipeline),
        Cint,
        (Csize_t, Cstring, Cstring, Ptr{UInt32}, Csize_t),
        Csize_t(id),
        wgsl_source,
        entry_point,
        binding_flags,
        Csize_t(length(binding_flags)),
    )
    _check_wgpu_status(status, "wgpu_create_compute_pipeline!")
    return nothing
end

"""
    wgpu_bind_buffers!(pipeline_id, buffer_ids)

Bind GPU buffers to pipeline `pipeline_id` in binding-index order.
`buffer_ids` is a `Vector{Int}` matching the `@binding(n)` slots in the shader.
Must be called after both pipeline and all buffers are created.
"""
function wgpu_bind_buffers!(pipeline_id::Integer, buffer_ids::Vector{<:Integer})
    ids = Csize_t[Csize_t(x) for x in buffer_ids]
    status = ccall(
        _axis_rs_symbol(:rust_wgpu_bind_buffers),
        Cint,
        (Csize_t, Ptr{Csize_t}, Csize_t),
        Csize_t(pipeline_id), ids, Csize_t(length(ids)),
    )
    _check_wgpu_status(status, "wgpu_bind_buffers!")
    return nothing
end

"""
    wgpu_dispatch!(pipeline_id; wg_x, wg_y=1, wg_z=1)

Dispatch compute pipeline `pipeline_id` with the given workgroup counts.
"""
function wgpu_dispatch!(pipeline_id::Integer; wg_x::Integer, wg_y::Integer=1, wg_z::Integer=1)
    status = ccall(
        _axis_rs_symbol(:rust_wgpu_dispatch),
        Cint,
        (Csize_t, UInt32, UInt32, UInt32),
        Csize_t(pipeline_id), UInt32(wg_x), UInt32(wg_y), UInt32(wg_z),
    )
    _check_wgpu_status(status, "wgpu_dispatch!")
    return nothing
end

"""
    wgpu_destroy_pipeline!(id)

Destroy compute pipeline `id` and free its GPU resources.
"""
function wgpu_destroy_pipeline!(id::Integer)
    status = ccall(
        _axis_rs_symbol(:rust_wgpu_destroy_pipeline),
        Cint,
        (Csize_t,),
        Csize_t(id),
    )
    _check_wgpu_status(status, "wgpu_destroy_pipeline!")
    return nothing
end

"""
    wgpu_read_buffer_and_broadcast!(id, path, byte_len)

Read GPU buffer `id` results, wrap it as Envelope V1, and broadcast via registered callback.
"""
function wgpu_read_buffer_and_broadcast!(id::Integer, path::String, byte_len::Integer)
    status = ccall(
        _axis_rs_symbol(:rust_wgpu_read_buffer_and_broadcast),
        Cint,
        (Csize_t, Cstring, Csize_t),
        Csize_t(id), path, Csize_t(byte_len),
    )
    _check_wgpu_status(status, "wgpu_read_buffer_and_broadcast!")
    return nothing
end

"""
    wgpu_dispatch_and_read_broadcast!(pipeline_id, buffer_id, path, byte_len; wg_x, wg_y, wg_z)

Unified dispatch, readback, and broadcast in a single CommandEncoder submission.
"""
function wgpu_dispatch_and_read_broadcast!(pipeline_id::Integer, buffer_id::Integer, path::String, byte_len::Integer;
                                           wg_x::Integer = 1, wg_y::Integer = 1, wg_z::Integer = 1)
    status = ccall(
        _axis_rs_symbol(:rust_wgpu_dispatch_and_read_broadcast),
        Cint,
        (Csize_t, Csize_t, UInt32, UInt32, UInt32, Cstring, Csize_t),
        Csize_t(pipeline_id), Csize_t(buffer_id), UInt32(wg_x), UInt32(wg_y), UInt32(wg_z), path, Csize_t(byte_len),
    )
    _check_wgpu_status(status, "wgpu_dispatch_and_read_broadcast!")
    return nothing
end

"""
    axis_set_broadcast_callback(ptr)

Set the global callback for Fomalhaut websocket broadcast.
"""
function axis_set_broadcast_callback(callback_ptr::Ptr{Cvoid})
    status = ccall(
        _axis_rs_symbol(:rust_axis_set_broadcast_callback),
        Cint,
        (Ptr{Cvoid},),
        callback_ptr,
    )
    _check_wgpu_status(status, "axis_set_broadcast_callback")
    return nothing
end

