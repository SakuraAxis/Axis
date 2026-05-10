use crate::ocean::{phillips_ocean, phillips_ocean_wgpu};
use std::{panic, slice};

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

#[unsafe(no_mangle)]
pub extern "C" fn rust_init_phillips_ocean_wgpu() -> i32 {
    match panic::catch_unwind(phillips_ocean_wgpu::init) {
        Ok(Ok(())) => 0,
        Ok(Err(phillips_ocean_wgpu::WgpuError::NoAdapter)) => -11,
        Ok(Err(phillips_ocean_wgpu::WgpuError::RequestDeviceFailed)) => -12,
        Ok(Err(_)) => -13,
        Err(_) => -17,
    }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn rust_upload_phillips_ocean_buffers(
    phase_base: *const f32,
    omega: *const f32,
    amp: *const f32,
    phase0: *const f32,
    frame_count: usize,
    component_count: usize,
) -> i32 {
    if phase_base.is_null() || omega.is_null() || amp.is_null() || phase0.is_null() {
        return -1;
    }

    let Some(phase_base_len) = frame_count.checked_mul(component_count) else {
        return -4;
    };

    let phase_base = unsafe { slice::from_raw_parts(phase_base, phase_base_len) };
    let omega = unsafe { slice::from_raw_parts(omega, component_count) };
    let amp = unsafe { slice::from_raw_parts(amp, component_count) };
    let phase0 = unsafe { slice::from_raw_parts(phase0, component_count) };

    let result = panic::catch_unwind(panic::AssertUnwindSafe(|| {
        phillips_ocean_wgpu::upload_buffers(
            phase_base,
            omega,
            amp,
            phase0,
            frame_count,
            component_count,
        )
    }));

    match result {
        Ok(Ok(())) => 0,
        Ok(Err(phillips_ocean_wgpu::WgpuError::NotInitialized)) => -10,
        Ok(Err(phillips_ocean_wgpu::WgpuError::InvalidFrameCount)) => -2,
        Ok(Err(phillips_ocean_wgpu::WgpuError::InvalidComponentCount)) => -3,
        Ok(Err(phillips_ocean_wgpu::WgpuError::BufferTooSmall)) => -4,
        Ok(Err(_)) => -13,
        Err(_) => -17,
    }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn rust_compute_phillips_ocean_wave(
    frame: *mut f32,
    frame_count: usize,
    time: f32,
) -> i32 {
    if frame.is_null() {
        return -1;
    }

    let frame = unsafe { slice::from_raw_parts_mut(frame, frame_count) };

    let result = panic::catch_unwind(panic::AssertUnwindSafe(|| {
        phillips_ocean_wgpu::compute_wave(frame, frame_count, time)
    }));

    match result {
        Ok(Ok(())) => 0,
        Ok(Err(phillips_ocean_wgpu::WgpuError::NotInitialized)) => -10,
        Ok(Err(phillips_ocean_wgpu::WgpuError::BuffersNotUploaded)) => -18,
        Ok(Err(phillips_ocean_wgpu::WgpuError::FrameCountMismatch)) => -19,
        Ok(Err(phillips_ocean_wgpu::WgpuError::InvalidFrameCount)) => -2,
        Ok(Err(phillips_ocean_wgpu::WgpuError::BufferTooSmall)) => -4,
        Ok(Err(phillips_ocean_wgpu::WgpuError::MapFailed)) => -14,
        Ok(Err(phillips_ocean_wgpu::WgpuError::PollFailed)) => -15,
        Ok(Err(phillips_ocean_wgpu::WgpuError::ReadbackFailed)) => -16,
        Ok(Err(_)) => -13,
        Err(_) => -17,
    }
}
