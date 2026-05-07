use crate::ocean::phillips_ocean;

#[unsafe(no_mangle)]
pub extern "C" fn rust_phillips_spectrum(
    kx: f32,
    ky: f32,
    wind_x: f32,
    wind_y: f32,
    wind_speed: f32,
    gravity: f32,
) -> f32 {
    phillips_ocean::phillips_spectrum(kx, ky, wind_x, wind_y, wind_speed, gravity)
}
