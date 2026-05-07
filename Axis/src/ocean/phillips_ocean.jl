const GRAVITY = 9.81f0
const WIND_SPEED = 14.0f0

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
