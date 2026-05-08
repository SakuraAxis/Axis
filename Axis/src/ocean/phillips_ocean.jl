const RESOLUTION = 96
const FRAME_INTERVAL = 1 / 30
const DOMAIN_SIZE = 36.0f0
const COMPONENT_COUNT = 128

const GRAVITY = 9.81f0
const WIND_SPEED = 14.0f0
const WIND_DIRECTION = (0.92f0, 0.38f0)
const AMPLITUDE_SCALE = 0.08f0

const KX = Vector{Float32}(undef, COMPONENT_COUNT)
const KY = Vector{Float32}(undef, COMPONENT_COUNT)
const OMEGA = Vector{Float32}(undef, COMPONENT_COUNT)
const AMP = Vector{Float32}(undef, COMPONENT_COUNT)
const PHASE0 = Vector{Float32}(undef, COMPONENT_COUNT)

"""
    phillips_spectrum(kx, ky, windx, windy; wind_speed=WIND_SPEED, gravity=GRAVITY)

Compute the Phillips ocean spectrum through the Rust implementation.
"""
function phillips_spectrum(
    kx::Float32,
    ky::Float32,
    windx::Float32,
    windy::Float32;
    wind_speed::Float32 = WIND_SPEED,
    gravity::Float32 = GRAVITY,
)
    return ccall(
        _axis_rs_symbol(:rust_phillips_spectrum),
        Cfloat,
        (Cfloat, Cfloat, Cfloat, Cfloat, Cfloat, Cfloat),
        kx,
        ky,
        windx,
        windy,
        wind_speed,
        gravity,
    )
end

function _check_component_buffer(name::String, data::Vector{Float32}, component_count::Integer)
    length(data) >= component_count && return nothing
    error("$name length must be at least $component_count, got $(length(data)).")
end

function _check_build_components_status(status::Cint)
    status == 0 && return nothing
    status == -1 && error("Rust build_components received a null pointer.")
    status == -2 && error("component_count must be a positive even number.")
    status == -3 && error("one or more component buffers are too small.")
    error("Rust build_components failed with status $status.")
end

"""
    build_components!(kx, ky, omega, amp, phase0; ...)

Fill Phillips ocean component buffers through the Rust implementation.
"""
function build_components!(
    kx::Vector{Float32},
    ky::Vector{Float32},
    omega::Vector{Float32},
    amp::Vector{Float32},
    phase0::Vector{Float32};
    component_count::Integer = COMPONENT_COUNT,
    wind_direction::Tuple{<:Real, <:Real} = WIND_DIRECTION,
    wind_speed::Real = WIND_SPEED,
    gravity::Real = GRAVITY,
    amplitude_scale::Real = AMPLITUDE_SCALE,
    seed::Integer = 42,
)
    _check_component_buffer("kx", kx, component_count)
    _check_component_buffer("ky", ky, component_count)
    _check_component_buffer("omega", omega, component_count)
    _check_component_buffer("amp", amp, component_count)
    _check_component_buffer("phase0", phase0, component_count)

    status = ccall(
        _axis_rs_symbol(:rust_build_phillips_ocean_components),
        Cint,
        (
            Ptr{Cfloat},
            Ptr{Cfloat},
            Ptr{Cfloat},
            Ptr{Cfloat},
            Ptr{Cfloat},
            Csize_t,
            Cfloat,
            Cfloat,
            Cfloat,
            Cfloat,
            Cfloat,
            UInt64,
        ),
        kx,
        ky,
        omega,
        amp,
        phase0,
        Csize_t(component_count),
        Float32(wind_direction[1]),
        Float32(wind_direction[2]),
        Float32(wind_speed),
        Float32(gravity),
        Float32(amplitude_scale),
        UInt64(seed),
    )

    _check_build_components_status(status)
    return (kx = kx, ky = ky, omega = omega, amp = amp, phase0 = phase0)
end

function build_components!(; kwargs...)
    return build_components!(KX, KY, OMEGA, AMP, PHASE0; kwargs...)
end

function phillips_spectrum(
    kx::Real,
    ky::Real,
    windx::Real,
    windy::Real;
    wind_speed::Real = WIND_SPEED,
    gravity::Real = GRAVITY,
)
    return phillips_spectrum(
        Float32(kx),
        Float32(ky),
        Float32(windx),
        Float32(windy);
        wind_speed = Float32(wind_speed),
        gravity = Float32(gravity),
    )
end
