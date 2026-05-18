#= 
OceanSim struct-based implementation

Each OceanSim instance owns its own CPU buffers and GPU resource IDs,
allowing multiple independent simulations to run simultaneously.

Public API :
- create_sim(; resolution, component_count, ...) -> OceanSim
- init!(sim)
- compute_wave!(sim, t) / compute_wave!(sim, output, t)
- update_params!(sim; wind_speed, wind_direction, amplitude_scale)
- destroy!(sim)
- phillips_spectrum(kx, ky, windx, windy)
=#

#= 
Unique GPU resource ID allocation ( thread- safe monotonic counter )
Buffers and Pipelines live in separate HashMaps in Rust, so IDs can overlap.
Convention : sim N gets buffer IDs ( 4N-3 )..( 4N ) and pipeline ID N.
=#
const _SIM_COUNTER = Ref{Int}(0)

function _alloc_sim_resources()
    id = (_SIM_COUNTER[] += 1)
    base = id * 4
    return (
        buf_frame      = base - 3,
        buf_components = base - 2,
        buf_phase0     = base - 1,
        buf_params     = base,
        pipeline_id    = id,
    )
end

#= 
PhillipsSim - holds all state for one independent ocean simulation
=#
mutable struct PhillipsSim <: AbstractOceanSim
    # Physics parameters
    resolution::Int
    component_count::Int
    domain_size::Float32
    gravity::Float32
    wind_speed::Float32
    wind_direction::NTuple{2, Float32}
    amplitude_scale::Float32
    frame_interval::Float32
    seed::UInt64

    # Pre-allocated CPU buffers ( zero GC in the hot path )
    kx::Vector{Float32}
    ky::Vector{Float32}
    omega::Vector{Float32}
    amp::Vector{Float32}
    phase0::Vector{Float32}
    
    frame_buffer::Vector{Float32}

    # Reusable scratch buffers ( avoids alloc on parameter updates )
    _params_buf::Vector{UInt8}
    _components_buf::Vector{Float32}

    # GPU resource IDs ( unique per instance, keys into Rust's HashMaps )
    _buf_frame::Int
    _buf_components::Int
    _buf_phase0::Int
    _buf_params::Int
    _pipeline_id::Int

    # Lifecycle state
    _gpu_ready::Bool
end

#= 
Constructor
=#
"""
    create_phillips_sim(; resolution, component_count, domain_size, gravity,
                          wind_speed, wind_direction, amplitude_scale,
                          frame_interval, seed) -> PhillipsSim

Allocate a new Phillips ocean simulation. CPU buffers are pre-allocated here;
GPU resources are created lazily during `init!(sim)`.
"""
function create_phillips_sim(;
    resolution::Int                         = 96,
    component_count::Int                    = 128,
    domain_size::Real                       = 36.0,
    gravity::Real                           = 9.81,
    wind_speed::Real                        = 14.0,
    wind_direction::Tuple{<:Real, <:Real}   = (0.92, 0.38),
    amplitude_scale::Real                   = 0.08,
    frame_interval::Real                    = 1 / 30,
    seed::Integer                           = 42,
)
    ids = _alloc_sim_resources()
    ds  = Float32(domain_size)
    n   = resolution

    return PhillipsSim(
        resolution, component_count,
        ds, Float32(gravity), Float32(wind_speed),
        (Float32(wind_direction[1]), Float32(wind_direction[2])),
        Float32(amplitude_scale), Float32(frame_interval), UInt64(seed),
        Vector{Float32}(undef, component_count), # kx
        Vector{Float32}(undef, component_count), # ky
        Vector{Float32}(undef, component_count), # omega
        Vector{Float32}(undef, component_count), # amp
        Vector{Float32}(undef, component_count), # phase0
        Vector{Float32}(undef, n * n), # frame_buffer
        Vector{UInt8}(undef, 16), # _params_buf
        Vector{Float32}(undef, component_count * 4), # _components_buf
        ids.buf_frame, ids.buf_components, ids.buf_phase0, ids.buf_params,
        ids.pipeline_id,
        false,
    )
end

#= 
WGSL compute shader ( Phillips Ocean kernel )
=#
const _OCEAN_WGSL = """
struct Params {
    resolution      : u32,
    component_count : u32,
    time            : f32,
    domain_size     : f32,
}

@group(0) @binding(0) var<storage, read_write> frame      : array<f32>;
@group(0) @binding(1) var<storage, read>       components : array<vec4<f32>>; // (kx, ky, amp, dynamic_phase)
@group(0) @binding(2) var<uniform>             params     : Params;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id : vec3<u32>) {
    let idx = global_id.x;
    let res = params.resolution;
    if (idx >= res * res) { return; }

    let ix = idx % res;
    let iy = idx / res;
    let x = (f32(ix) / f32(res - 1u) - 0.5) * params.domain_size;
    let y = (f32(iy) / f32(res - 1u) - 0.5) * params.domain_size;

    var height = 0.0;
    for (var c = 0u; c < params.component_count; c = c + 1u) {
        let comp = components[c];
        let phase = comp.x * x + comp.y * y + comp.w;
        height = height + comp.z * cos(phase);
    }
    frame[idx] = height;
}
"""

const _WORKGROUP_SIZE = 256

#= 
CPU math helpers
=#

"""
    phillips_spectrum(kx, ky, windx, windy; wind_speed, gravity) -> Float32

Pure-Julia Phillips ocean spectrum — no FFI, no allocation.
"""
function phillips_spectrum(
    kx::Float32, ky::Float32, windx::Float32, windy::Float32;
    wind_speed::Float32 = 14.0f0,
    gravity::Float32    = 9.81f0,
)::Float32
    k2 = kx * kx + ky * ky
    k2 < 1f-6 && return 0f0
    k         = sqrt(k2)
    alignment = max((kx / k) * windx + (ky / k) * windy, 0f0)
    L         = (wind_speed * wind_speed) / gravity
    l2_small  = (L * 0.0015f0)^2
    exp(-1f0 / (k2 * L * L)) / (k2 * k2) * alignment^4 * exp(-k2 * l2_small)
end

phillips_spectrum(kx::Real, ky::Real, windx::Real, windy::Real; kwargs...) =
    phillips_spectrum(Float32(kx), Float32(ky), Float32(windx), Float32(windy); kwargs...)

#= 
Minimal xorshift64 RNG ( matches Rust AxisRng for identical numeric output )
=#
mutable struct _AxisRng; state::UInt64; end
_AxisRng(seed::Integer) = _AxisRng(UInt64(max(seed, 1)))

function _next_u32!(r::_AxisRng)::UInt32
    x = r.state
    x ⊻= x >> 12; x ⊻= x << 25; x ⊻= x >> 27
    r.state = x
    UInt32((x * 0x2545f4914f6cdd1d % typemax(UInt64)) >> 32)
end

_next_f32!(r::_AxisRng)::Float32 =
    Float32(_next_u32!(r) >> 8) * (1f0 / Float32(1 << 24))

function _std_normal!(r::_AxisRng)::Float32
    u1 = max(_next_f32!(r), floatmin(Float32))
    sqrt(-2f0 * log(u1)) * cos(2f0 * Float32(π) * _next_f32!(r))
end

#= 
Component construction
=#

"""
    build_components!(sim)

Fill `sim`'s SoA wave-component buffers (kx, ky, omega, amp, phase0).
Pure Julia - zero allocation ( all output arrays are pre-allocated on the sim ).
"""
function build_components!(sim::PhillipsSim)
    cc   = sim.component_count
    rng  = _AxisRng(sim.seed)
    wxn, wyn = normalize2(sim.wind_direction[1], sim.wind_direction[2])
    base_angle = atan(wyn, wxn)
    pair_count = cc ÷ 2
    idx = 1

    @inbounds for i in 0:(pair_count - 1)
        band  = pair_count <= 1 ? 0f0 : Float32(i) / Float32(pair_count - 1)
        k     = 2f0 * Float32(π) / (1.2f0 + 9.0f0 * band^2)
        angle = base_angle + _std_normal!(rng) * 1.05f0 * (0.2f0 + 0.8f0 * band)

        for (dir, scale) in ((1f0, 1f0), (-1f0, 0.45f0))
            wkx = dir * cos(angle) * k
            wky = dir * sin(angle) * k
            spec = phillips_spectrum(wkx, wky, wxn, wyn;
                                     wind_speed = sim.wind_speed,
                                     gravity    = sim.gravity)
            sim.amp[idx]    = sim.amplitude_scale * scale *
                              sqrt(max(spec, 0f0)) * (0.35f0 + 0.65f0 * (1f0 - band))
            sim.phase0[idx] = _next_f32!(rng) * 2f0 * Float32(π)
            sim.omega[idx]  = sqrt(sim.gravity * k)
            sim.kx[idx]     = wkx
            sim.ky[idx]     = wky
            idx += 1
        end
    end
    return sim
end

#= 
GPU buffer packing helpers
=#
function _pack_components!(sim::PhillipsSim, t::Float32 = 0f0)
    cc  = sim.component_count
    buf = sim._components_buf
    @inbounds for i in 1:cc
        b = (i - 1) * 4 + 1
        buf[b]     = sim.kx[i]
        buf[b + 1] = sim.ky[i]
        buf[b + 2] = sim.amp[i]
        buf[b + 3] = sim.phase0[i] - sim.omega[i] * t
    end
    return buf
end

function _write_params!(sim::PhillipsSim, t::Float32)
    buf = sim._params_buf        # reuse pre-allocated 16-byte scratch — no alloc
    buf[1:4]   .= reinterpret(UInt8, [UInt32(sim.resolution)])
    buf[5:8]   .= reinterpret(UInt8, [UInt32(sim.component_count)])
    buf[9:12]  .= reinterpret(UInt8, [t])
    buf[13:16] .= reinterpret(UInt8, [sim.domain_size])
    return buf
end

#= 
GPU upload & Update
=#
"""
    upload_buffers!(sim)

Create GPU buffers, compile the WGSL shader, and bind everything.
Called automatically by `init!(sim)`.
"""
function upload_buffers!(sim::PhillipsSim)
    fc  = sim.resolution * sim.resolution
    cc  = sim.component_count

    wgpu_create_buffer!(sim._buf_frame,      fc * 4,        BINDING_STORAGE_READ_WRITE)
    wgpu_create_buffer!(sim._buf_components, cc * 4 * 4,    BINDING_STORAGE_READ)
    wgpu_create_buffer!(sim._buf_params,     16,            BINDING_UNIFORM)

    wgpu_write_buffer!(sim._buf_components, _pack_components!(sim, 0f0))
    wgpu_write_buffer!(sim._buf_params,     _write_params!(sim, 0f0))

    binding_flags = UInt32[
        BINDING_STORAGE_READ_WRITE,
        BINDING_STORAGE_READ,
        BINDING_UNIFORM,
    ]
    wgpu_create_compute_pipeline!(sim._pipeline_id, _OCEAN_WGSL, "main", binding_flags)
    wgpu_bind_buffers!(sim._pipeline_id,
        [sim._buf_frame, sim._buf_components, sim._buf_params])

    sim._gpu_ready = true
    return sim
end

"""
    update_params!(sim; kwargs...)

Update wave generation parameters (wind, amplitude) dynamically, recalculate
components on CPU, and upload changes directly to GPU buffers with zero allocation.
"""
function update_params!(sim::PhillipsSim; 
    wind_speed::Union{Real, Nothing} = nothing,
    wind_direction::Union{Tuple{<:Real, <:Real}, Nothing} = nothing,
    amplitude_scale::Union{Real, Nothing} = nothing)
    
    changed = false
    if wind_speed !== nothing
        sim.wind_speed = Float32(wind_speed)
        changed = true
    end
    if wind_direction !== nothing
        sim.wind_direction = (Float32(wind_direction[1]), Float32(wind_direction[2]))
        changed = true
    end
    if amplitude_scale !== nothing
        sim.amplitude_scale = Float32(amplitude_scale)
        changed = true
    end
    
    if changed && sim._gpu_ready
        build_components!(sim)
        wgpu_write_buffer!(sim._buf_components, _pack_components!(sim))
    end
    return sim
end


#= 
Full init
=#
"""
    init!(sim)

Run the full initialisation sequence:
  1. Start WGPU dispatcher ( idempotent )
  2. Build wave components ( CPU )
  3. Upload buffers and compile WGSL shader ( GPU )
"""
function init!(sim::PhillipsSim)
    wgpu_init!()
    build_components!(sim)
    upload_buffers!(sim)
    @info "PhillipsOcean Initialized" backend="wgpu ( Axis )"
    return sim
end

#= 
Per-frame compute
=#
"""
    compute_wave!(sim, output, t)

Dispatch the GPU ocean kernel for time `t`, readback directly into `output`.
No heap allocation in the hot path.
"""
function compute_wave!(sim::PhillipsSim, output::Vector{Float32}, t::Real)
    sim._gpu_ready || error("OceanSim not initialized — call init!(sim) first.")
    fc = sim.resolution * sim.resolution

    wgpu_write_buffer!(sim._buf_components, _pack_components!(sim, Float32(t)))
    wgpu_write_buffer!(sim._buf_params, _write_params!(sim, Float32(t)))
    wgpu_dispatch!(sim._pipeline_id; wg_x = cld(fc, _WORKGROUP_SIZE))
    wgpu_read_buffer!(sim._buf_frame, output)

    return output
end

"""
    compute_wave!(sim, t)

Convenience overload — writes into `sim.frame_buffer`.
"""
compute_wave!(sim::PhillipsSim, t::Real) = compute_wave!(sim, sim.frame_buffer, t)

"""
    compute_wave_and_broadcast!(sim, t, path)

Dispatch the GPU ocean kernel for time `t`, readback directly in Rust, package as Envelope V1, and broadcast via registered callback to Fomalhaut.
No heap allocation in the hot path.
"""
function compute_wave_and_broadcast!(sim::PhillipsSim, t::Real, path::String)
    sim._gpu_ready || error("OceanSim not initialized — call init!(sim) first.")
    fc = sim.resolution * sim.resolution

    wgpu_write_buffer!(sim._buf_components, _pack_components!(sim, Float32(t)))
    wgpu_write_buffer!(sim._buf_params, _write_params!(sim, Float32(t)))
    wgpu_dispatch_and_read_broadcast!(sim._pipeline_id, sim._buf_frame, path, fc * 4; wg_x = cld(fc, _WORKGROUP_SIZE))

    return nothing
end

#= 
Cleanup
=#

"""
    destroy!(sim)

Free all GPU resources associated with this simulation.
The `sim` object must not be used after calling this.
"""
function destroy!(sim::PhillipsSim)
    sim._gpu_ready || return
    wgpu_destroy_pipeline!(sim._pipeline_id)
    wgpu_destroy_buffer!(sim._buf_frame)
    wgpu_destroy_buffer!(sim._buf_components)
    wgpu_destroy_buffer!(sim._buf_params)
    sim._gpu_ready = false
    return nothing
end
