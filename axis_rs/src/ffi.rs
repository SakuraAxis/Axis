use crate::wgpu_core;
use std::{os::raw::c_char, panic};

// Helper : wrap a fallible closure, catching panics, returning i32 error code
macro_rules! ffi_call {
    ($body:expr) => {
        match panic::catch_unwind(panic::AssertUnwindSafe(|| $body)) {
            Ok(code) => code,
            Err(_) => wgpu_core::ERR_PANIC,
        }
    };
}

// Dispatcher lifecycle

/// Initialise the WGPU device and dispatcher.
#[unsafe(no_mangle)]
pub extern "C" fn rust_wgpu_init() -> i32 {
    ffi_call!(wgpu_core::init())
}

// Buffer management

/// Create a GPU buffer.
/// `binding_type`: 0=storage_read, 1=storage_read_write, 2=uniform
#[unsafe(no_mangle)]
pub extern "C" fn rust_wgpu_create_buffer(id: usize, size: usize, binding_type: u32) -> i32 {
    ffi_call!(wgpu_core::create_buffer(id, size, binding_type))
}

/// Write `byte_len` bytes from `data_ptr` into GPU buffer `id`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rust_wgpu_write_buffer(
    id: usize,
    data_ptr: *const u8,
    byte_len: usize,
) -> i32 {
    ffi_call!(wgpu_core::write_buffer(id, data_ptr, byte_len))
}

/// Read `byte_len` bytes from GPU buffer `id` back to `data_ptr` ( CPU ).
/// Only valid for read-write storage buffers.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rust_wgpu_read_buffer(
    id: usize,
    data_ptr: *mut u8,
    byte_len: usize,
) -> i32 {
    ffi_call!(wgpu_core::read_buffer(id, data_ptr, byte_len))
}

/// Destroy GPU buffer `id` and free its memory.
#[unsafe(no_mangle)]
pub extern "C" fn rust_wgpu_destroy_buffer(id: usize) -> i32 {
    ffi_call!(wgpu_core::destroy_buffer(id))
}

// Pipeline management

/// Compile and register a WGSL compute pipeline.
///
/// `wgsl_ptr`           - null-terminated UTF-8 WGSL source string
/// `entry_ptr`          - null-terminated entry-point name ( e.g. "main" )
/// `binding_flags_ptr`  - C array of `binding_count` u32 binding-type flags
///                        ( 0=storage_read, 1=storage_read_write, 2=uniform )
/// `binding_count`      - number of @binding entries in the shader
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rust_wgpu_create_compute_pipeline(
    id: usize,
    wgsl_ptr: *const c_char,
    entry_ptr: *const c_char,
    binding_flags_ptr: *const u32,
    binding_count: usize,
) -> i32 {
    ffi_call!(wgpu_core::create_compute_pipeline(
        id,
        wgsl_ptr,
        entry_ptr,
        binding_flags_ptr,
        binding_count,
    ))
}

/// Bind GPU buffers to a pipeline ( in binding-index order ).
/// Call after both `rust_wgpu_create_compute_pipeline` and `rust_wgpu_create_buffer`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rust_wgpu_bind_buffers(
    pipeline_id: usize,
    buffer_ids_ptr: *const usize,
    buffer_count: usize,
) -> i32 {
    ffi_call!(wgpu_core::bind_buffers(
        pipeline_id,
        buffer_ids_ptr,
        buffer_count,
    ))
}

/// Dispatch a compute pipeline with the given workgroup counts.
#[unsafe(no_mangle)]
pub extern "C" fn rust_wgpu_dispatch(
    pipeline_id: usize,
    wg_x: u32,
    wg_y: u32,
    wg_z: u32,
) -> i32 {
    ffi_call!(wgpu_core::dispatch(pipeline_id, wg_x, wg_y, wg_z))
}

/// Destroy a compute pipeline and free its GPU resources.
#[unsafe(no_mangle)]
pub extern "C" fn rust_wgpu_destroy_pipeline(id: usize) -> i32 {
    ffi_call!(wgpu_core::destroy_pipeline(id))
}

