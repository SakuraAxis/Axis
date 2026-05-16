#= 
OceanSim struct-based implementation

Each OceanSim instance owns its own CPU buffers and GPU resource IDs,
allowing multiple independent simulations to run simultaneously.

Public API :
- create_sim(; resolution, component_count, ...) -> OceanSim
- init!(sim)
- compute_wave!(sim, t) / compute_wave!(sim, output, t)
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
        buf_phase_base = base - 2,
        buf_components = base - 1,
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
    phase_base::Matrix{Float32}
    frame_buffer::Vector{Float32}
    grid_x::Matrix{Float32}
    grid_y::Matrix{Float32}

    # Reusable params scratch buffer ( avoids 16-byte alloc every frame )
    _params_buf::Vector{UInt8}

    # GPU resource IDs ( unique per instance, keys into Rust's HashMaps )
    _buf_frame::Int
    _buf_phase_base::Int
    _buf_components::Int
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

    grid_x = Float32[((x - 1) / (n - 1) - 0.5f0) * ds for _ in 1:n, x in 1:n]
    grid_y = Float32[((y - 1) / (n - 1) - 0.5f0) * ds for y in 1:n, _ in 1:n]

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
        Matrix{Float32}(undef, n * n, component_count), # phase_base
        Vector{Float32}(undef, n * n), # frame_buffer
        grid_x, grid_y,
        Vector{UInt8}(undef, 16), # _params_buf
        ids.buf_frame, ids.buf_phase_base, ids.buf_components, ids.buf_params,
        ids.pipeline_id,
        false,
    )
end

#= 
WGSL compute shader ( Phillips Ocean kernel )
=#
const _OCEAN_WGSL = """
struct Params {
    frame_count     : u32,
    component_count : u32,
    time            : f32,
    _pad            : f32,
}

@group(0) @binding(0) var<storage, read_write> frame      : array<f32>;
@group(0) @binding(1) var<storage, read>       phase_base : array<f32>;
@group(0) @binding(2) var<storage, read>       components : array<vec4<f32>>;
@group(0) @binding(3) var<uniform>             params     : Params;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id : vec3<u32>) {
    let idx = global_id.x;
    if (idx >= params.frame_count) { return; }

    var height = 0.0;
    for (var c = 0u; c < params.component_count; c = c + 1u) {
        let data = components[c];
        height   = height + data.y *
            cos(phase_base[idx + c * params.frame_count] - data.x * params.time + data.z);
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

"""
    precompute_phase!(sim)

Fill `sim.phase_base` with `kx·x + ky·y` for every grid point and component.
"""
function precompute_phase!(sim::PhillipsSim)
    fc  = length(sim.grid_x)
    cc  = sim.component_count
    @inbounds for c in 1:cc
        kxc = sim.kx[c]; kyc = sim.ky[c]
        for i in 1:fc
            sim.phase_base[i, c] = kxc * sim.grid_x[i] + kyc * sim.grid_y[i]
        end
    end
    return sim
end

#= 
GPU buffer packing helpers
=#
function _pack_components!(sim::PhillipsSim)
    cc   = sim.component_count
    data = Vector{Float32}(undef, cc * 4)
    @inbounds for i in 1:cc
        b = (i - 1) * 4 + 1
        data[b]     = sim.omega[i]
        data[b + 1] = sim.amp[i]
        data[b + 2] = sim.phase0[i]
        data[b + 3] = 0f0
    end
    return data
end

function _write_params!(sim::PhillipsSim, frame_count::Int, t::Float32)
    buf = sim._params_buf        # reuse pre-allocated 16-byte scratch — no alloc
    buf[1:4]   .= reinterpret(UInt8, [UInt32(frame_count)])
    buf[5:8]   .= reinterpret(UInt8, [UInt32(sim.component_count)])
    buf[9:12]  .= reinterpret(UInt8, [t])
    buf[13:16] .= reinterpret(UInt8, [0f0])
    return buf
end

#= 
GPU upload
=#
"""
    upload_buffers!(sim)

Create GPU buffers, compile the WGSL shader, and bind everything.
Called automatically by `init!(sim)`.
"""
function upload_buffers!(sim::PhillipsSim)
    fc  = length(sim.frame_buffer)
    cc  = sim.component_count

    wgpu_create_buffer!(sim._buf_frame,      fc * 4,        BINDING_STORAGE_READ_WRITE)
    wgpu_create_buffer!(sim._buf_phase_base, fc * cc * 4,   BINDING_STORAGE_READ)
    wgpu_create_buffer!(sim._buf_components, cc * 4 * 4,    BINDING_STORAGE_READ)
    wgpu_create_buffer!(sim._buf_params,     16,            BINDING_UNIFORM)

    wgpu_write_buffer!(sim._buf_phase_base, sim.phase_base)
    wgpu_write_buffer!(sim._buf_components, _pack_components!(sim))
    wgpu_write_buffer!(sim._buf_params,     _write_params!(sim, fc, 0f0))

    binding_flags = UInt32[
        BINDING_STORAGE_READ_WRITE,
        BINDING_STORAGE_READ,
        BINDING_STORAGE_READ,
        BINDING_UNIFORM,
    ]
    wgpu_create_compute_pipeline!(sim._pipeline_id, _OCEAN_WGSL, "main", binding_flags)
    wgpu_bind_buffers!(sim._pipeline_id,
        [sim._buf_frame, sim._buf_phase_base, sim._buf_components, sim._buf_params])

    sim._gpu_ready = true
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
  3. Pre-compute phase grid ( CPU )
  4. Upload buffers and compile WGSL shader ( GPU )
"""
function init!(sim::PhillipsSim)
    wgpu_init!()
    build_components!(sim)
    precompute_phase!(sim)
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
    fc = length(output)

    wgpu_write_buffer!(sim._buf_params, _write_params!(sim, fc, Float32(t)))
    wgpu_dispatch!(sim._pipeline_id; wg_x = cld(fc, _WORKGROUP_SIZE))
    wgpu_read_buffer!(sim._buf_frame, output)

    return output
end

"""
    compute_wave!(sim, t)

Convenience overload — writes into `sim.frame_buffer`.
"""
compute_wave!(sim::PhillipsSim, t::Real) = compute_wave!(sim, sim.frame_buffer, t)

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
    wgpu_destroy_buffer!(sim._buf_phase_base)
    wgpu_destroy_buffer!(sim._buf_components)
    wgpu_destroy_buffer!(sim._buf_params)
    sim._gpu_ready = false
    return nothing
end
