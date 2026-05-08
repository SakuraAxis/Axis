use crate::ocean::phillips_ocean;
use std::slice;

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

#[allow(clippy::too_many_arguments)]
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rust_build_phillips_ocean_components(
    kx: *mut f32,
    ky: *mut f32,
    omega: *mut f32,
    amp: *mut f32,
    phase0: *mut f32,
    component_count: usize,
    wind_x: f32,
    wind_y: f32,
    wind_speed: f32,
    gravity: f32,
    amplitude_scale: f32,
    seed: u64,
) -> i32 {
    if kx.is_null() || ky.is_null() || omega.is_null() || amp.is_null() || phase0.is_null() {
        return -1;
    }

    let kx = unsafe { slice::from_raw_parts_mut(kx, component_count) };
    let ky = unsafe { slice::from_raw_parts_mut(ky, component_count) };
    let omega = unsafe { slice::from_raw_parts_mut(omega, component_count) };
    let amp = unsafe { slice::from_raw_parts_mut(amp, component_count) };
    let phase0 = unsafe { slice::from_raw_parts_mut(phase0, component_count) };

    match phillips_ocean::build_components(
        kx,
        ky,
        omega,
        amp,
        phase0,
        component_count,
        wind_x,
        wind_y,
        wind_speed,
        gravity,
        amplitude_scale,
        seed,
    ) {
        Ok(()) => 0,
        Err(phillips_ocean::BuildComponentsError::InvalidComponentCount) => -2,
        Err(phillips_ocean::BuildComponentsError::BufferTooSmall) => -3,
    }
}
