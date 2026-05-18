use std::{
    collections::HashMap,
    future::Future,
    pin::Pin,
    sync::{Mutex, OnceLock},
    task::{Context, Poll, Wake, Waker},
    thread,
};

// Binding type flags ( mirrors the values passed from Julia via C-ABI )
/// Storage buffer, read-only (flag = 0)
pub const BINDING_STORAGE_READ: u32 = 0;
/// Storage buffer, read-write (flag = 1)
pub const BINDING_STORAGE_READ_WRITE: u32 = 1;
/// Uniform buffer (flag = 2)
pub const BINDING_UNIFORM: u32 = 2;

// Fomalhaut broadcast callback type
type WsBroadcastFn = unsafe extern "C" fn(
    path_ptr: *const u8,
    path_len: usize,
    frame_ptr: *const u8,
    frame_len: usize,
) -> i32;

static BROADCAST_CALLBACK: std::sync::Mutex<Option<WsBroadcastFn>> = std::sync::Mutex::new(None);

// Error codes exposed through the C-ABI ( negative i32 )
pub const OK: i32 = 0;
pub const ERR_NULL_PTR: i32 = -1;
pub const ERR_NOT_INITIALIZED: i32 = -2;
pub const ERR_ALREADY_EXISTS: i32 = -3;
pub const ERR_NOT_FOUND: i32 = -4;
pub const ERR_INVALID_ARG: i32 = -5;
pub const ERR_NO_ADAPTER: i32 = -11;
pub const ERR_DEVICE_FAILED: i32 = -12;
pub const ERR_SHADER_INVALID: i32 = -13;
pub const ERR_MAP_FAILED: i32 = -14;
pub const ERR_POLL_FAILED: i32 = -15;
pub const ERR_READBACK_FAILED: i32 = -16;
pub const ERR_PANIC: i32 = -17;

// Minimal async block-on ( no tokio dependency )
struct ThreadWaker {
    thread: thread::Thread,
}

impl Wake for ThreadWaker {
    fn wake(self: std::sync::Arc<Self>) {
        self.thread.unpark();
    }
    fn wake_by_ref(self: &std::sync::Arc<Self>) {
        self.thread.unpark();
    }
}

pub(crate) fn block_on<F: Future>(future: F) -> F::Output {
    let waker = Waker::from(std::sync::Arc::new(ThreadWaker {
        thread: thread::current(),
    }));
    let mut ctx = Context::from_waker(&waker);
    let mut future = std::pin::pin!(future);
    loop {
        match Future::poll(Pin::as_mut(&mut future), &mut ctx) {
            Poll::Ready(v) => return v,
            Poll::Pending => thread::park(),
        }
    }
}

// Internal GPU buffer record
struct GpuBuffer {
    buffer: wgpu::Buffer,
    size: usize,
    /// Readback staging buffer, only allocated for read_write storage buffers
    readback: Option<wgpu::Buffer>,
}

// Internal pipeline record
struct GpuPipeline {
    pipeline: wgpu::ComputePipeline,
    bind_group_layout: wgpu::BindGroupLayout,
    /// Ordered list of (buffer_id, binding_index)
    bindings: Vec<(usize, u32)>,
    bind_group: Option<wgpu::BindGroup>,
}

// Global dispatcher state
struct WgpuDispatcher {
    device: wgpu::Device,
    queue: wgpu::Queue,
    buffers: HashMap<usize, GpuBuffer>,
    pipelines: HashMap<usize, GpuPipeline>,
}

static DISPATCHER: OnceLock<Mutex<Option<WgpuDispatcher>>> = OnceLock::new();

fn dispatcher_slot() -> &'static Mutex<Option<WgpuDispatcher>> {
    DISPATCHER.get_or_init(|| Mutex::new(None))
}

// Public API functions ( called from ffi.rs )

/// Initialise the WGPU device. Idempotent - safe to call multiple times.
pub fn init() -> i32 {
    let mut slot = match dispatcher_slot().lock() {
        Ok(s) => s,
        Err(_) => return ERR_DEVICE_FAILED,
    };

    if slot.is_some() {
        return OK; // already initialized
    }

    let instance = wgpu::Instance::default();
    let adapter = match block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
        power_preference: wgpu::PowerPreference::HighPerformance,
        force_fallback_adapter: false,
        compatible_surface: None,
    })) {
        Ok(a) => a,
        Err(_) => return ERR_NO_ADAPTER,
    };

    let (device, queue) = match block_on(adapter.request_device(&wgpu::DeviceDescriptor {
        label: Some("Axis Dispatcher Device"),
        required_features: wgpu::Features::empty(),
        required_limits: wgpu::Limits::downlevel_defaults(),
        experimental_features: wgpu::ExperimentalFeatures::disabled(),
        memory_hints: wgpu::MemoryHints::Performance,
        trace: wgpu::Trace::Off,
    })) {
        Ok(d) => d,
        Err(_) => return ERR_DEVICE_FAILED,
    };

    *slot = Some(WgpuDispatcher {
        device,
        queue,
        buffers: HashMap::new(),
        pipelines: HashMap::new(),
    });

    OK
}

/// Create a GPU buffer with a given id, byte-size, and binding-type flag.
/// binding_type: 0=storage_read, 1=storage_read_write, 2=uniform
pub fn create_buffer(id: usize, size: usize, binding_type: u32) -> i32 {
    if size == 0 {
        return ERR_INVALID_ARG;
    }

    let mut slot = match dispatcher_slot().lock() {
        Ok(s) => s,
        Err(_) => return ERR_NOT_INITIALIZED,
    };
    let dispatcher = match slot.as_mut() {
        Some(d) => d,
        None => return ERR_NOT_INITIALIZED,
    };

    if dispatcher.buffers.contains_key(&id) {
        return ERR_ALREADY_EXISTS;
    }

    let (usage, needs_readback) = match binding_type {
        BINDING_STORAGE_READ => (wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST, false),
        BINDING_STORAGE_READ_WRITE => (
            wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST,
            true,
        ),
        BINDING_UNIFORM => (wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST, false),
        _ => return ERR_INVALID_ARG,
    };

    let buffer = dispatcher.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some(&format!("Axis Buffer {id}")),
        size: size as wgpu::BufferAddress,
        usage,
        mapped_at_creation: false,
    });

    let readback = if needs_readback {
        Some(dispatcher.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(&format!("Axis Readback Buffer {id}")),
            size: size as wgpu::BufferAddress,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        }))
    } else {
        None
    };

    dispatcher.buffers.insert(id, GpuBuffer { buffer, size, readback });
    OK
}

/// Write raw bytes from a CPU pointer into a GPU buffer.
pub fn write_buffer(id: usize, data_ptr: *const u8, byte_len: usize) -> i32 {
    if data_ptr.is_null() {
        return ERR_NULL_PTR;
    }

    let data = unsafe { std::slice::from_raw_parts(data_ptr, byte_len) };

    let mut slot = match dispatcher_slot().lock() {
        Ok(s) => s,
        Err(_) => return ERR_NOT_INITIALIZED,
    };
    let dispatcher = match slot.as_mut() {
        Some(d) => d,
        None => return ERR_NOT_INITIALIZED,
    };

    let gpu_buf = match dispatcher.buffers.get(&id) {
        Some(b) => b,
        None => return ERR_NOT_FOUND,
    };

    if byte_len > gpu_buf.size {
        return ERR_INVALID_ARG;
    }

    dispatcher.queue.write_buffer(&gpu_buf.buffer, 0, data);
    OK
}

/// Read GPU buffer results back to a CPU pointer ( only for read-write storage buffers ).
pub fn read_buffer(id: usize, data_ptr: *mut u8, byte_len: usize) -> i32 {
    if data_ptr.is_null() {
        return ERR_NULL_PTR;
    }

    let slot = match dispatcher_slot().lock() {
        Ok(s) => s,
        Err(_) => return ERR_NOT_INITIALIZED,
    };
    let dispatcher = match slot.as_ref() {
        Some(d) => d,
        None => return ERR_NOT_INITIALIZED,
    };

    // Destructure for disjoint field borrows : `buffers` immutable + `device` + `queue`.
    let WgpuDispatcher {
        ref device,
        ref queue,
        ref buffers,
        ..
    } = *dispatcher;

    let gpu_buf = match buffers.get(&id) {
        Some(b) => b,
        None => return ERR_NOT_FOUND,
    };

    let readback = match gpu_buf.readback.as_ref() {
        Some(r) => r,
        None => return ERR_INVALID_ARG, // not a read_write buffer
    };

    if byte_len > gpu_buf.size {
        return ERR_INVALID_ARG;
    }

    // Copy GPU storage -> readback staging buffer
    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("Axis Readback Encoder"),
    });
    encoder.copy_buffer_to_buffer(&gpu_buf.buffer, 0, readback, 0, byte_len as wgpu::BufferAddress);
    queue.submit(Some(encoder.finish()));

    // Map and wait
    use std::sync::mpsc;
    let (tx, rx) = mpsc::channel();
    readback.slice(..byte_len as u64).map_async(wgpu::MapMode::Read, move |r| {
        let _ = tx.send(r);
    });

    if device.poll(wgpu::PollType::wait_indefinitely()).is_err() {
        return ERR_POLL_FAILED;
    }

    if rx.recv().map_err(|_| ()).and_then(|r| r.map_err(|_| ())).is_err() {
        return ERR_MAP_FAILED;
    }

    {
        let view = readback.slice(..byte_len as u64).get_mapped_range();
        let dst = unsafe { std::slice::from_raw_parts_mut(data_ptr, byte_len) };
        dst.copy_from_slice(&view);
    }
    readback.unmap();

    OK
}

/// Read GPU buffer results back, encode as Envelope V1, and broadcast via registered callback.
pub fn read_buffer_and_broadcast(id: usize, path_ptr: *const std::os::raw::c_char, byte_len: usize) -> i32 {
    if path_ptr.is_null() {
        return ERR_NULL_PTR;
    }

    let path = unsafe {
        match std::ffi::CStr::from_ptr(path_ptr).to_str() {
            Ok(s) => s.to_owned(),
            Err(_) => return ERR_INVALID_ARG,
        }
    };

    let slot = match dispatcher_slot().lock() {
        Ok(s) => s,
        Err(_) => return ERR_NOT_INITIALIZED,
    };
    let dispatcher = match slot.as_ref() {
        Some(d) => d,
        None => return ERR_NOT_INITIALIZED,
    };

    let WgpuDispatcher {
        ref device,
        ref queue,
        ref buffers,
        ..
    } = *dispatcher;

    let gpu_buf = match buffers.get(&id) {
        Some(b) => b,
        None => return ERR_NOT_FOUND,
    };

    let readback = match gpu_buf.readback.as_ref() {
        Some(r) => r,
        None => return ERR_INVALID_ARG,
    };

    if byte_len > gpu_buf.size {
        return ERR_INVALID_ARG;
    }

    // Copy GPU storage -> readback staging buffer
    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("Axis Readback Encoder"),
    });
    encoder.copy_buffer_to_buffer(&gpu_buf.buffer, 0, readback, 0, byte_len as wgpu::BufferAddress);
    queue.submit(Some(encoder.finish()));

    // Map and wait
    use std::sync::mpsc;
    let (tx, rx) = mpsc::channel();
    readback.slice(..byte_len as u64).map_async(wgpu::MapMode::Read, move |r| {
        let _ = tx.send(r);
    });

    if device.poll(wgpu::PollType::wait_indefinitely()).is_err() {
        return ERR_POLL_FAILED;
    }

    if rx.recv().map_err(|_| ()).and_then(|r| r.map_err(|_| ())).is_err() {
        return ERR_MAP_FAILED;
    }

    let broadcast_fn = {
        let guard = BROADCAST_CALLBACK.lock().unwrap();
        match *guard {
            Some(f) => f,
            None => {
                readback.unmap();
                return ERR_INVALID_ARG; // No callback registered
            }
        }
    };

    {
        let view = readback.slice(..byte_len as u64).get_mapped_range();
        
        // Pack directly into Envelope V1 format
        let mut frame = Vec::with_capacity(17 + byte_len);
        frame.push(1); // version = 1
        frame.extend_from_slice(&1u16.to_le_bytes()); // contentType = 1 (Float32 Tensor)
        frame.extend_from_slice(&0u16.to_le_bytes()); // flags = 0
        let ts = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos() as u64;
        frame.extend_from_slice(&ts.to_le_bytes()); // timestampNs
        frame.extend_from_slice(&(byte_len as u32).to_le_bytes()); // payloadLen
        frame.extend_from_slice(&view); // payload

        unsafe {
            let path_bytes = path.as_bytes();
            broadcast_fn(
                path_bytes.as_ptr(),
                path_bytes.len(),
                frame.as_ptr(),
                frame.len(),
            );
        }
    }
    readback.unmap();

    OK
}

/// Unified Dispatch, Readback, and Broadcast in a single CommandEncoder submission.
/// This reduces PCIe/driver overhead by eliminating redundant queue.submit() calls.
pub fn dispatch_and_read_broadcast(
    pipeline_id: usize,
    buffer_id: usize,
    wg_x: u32,
    wg_y: u32,
    wg_z: u32,
    path_ptr: *const std::os::raw::c_char,
    byte_len: usize,
) -> i32 {
    if path_ptr.is_null() {
        return ERR_NULL_PTR;
    }
    let path = unsafe {
        match std::ffi::CStr::from_ptr(path_ptr).to_str() {
            Ok(s) => s.to_owned(),
            Err(_) => return ERR_INVALID_ARG,
        }
    };

    let slot = match dispatcher_slot().lock() {
        Ok(s) => s,
        Err(_) => return ERR_NOT_INITIALIZED,
    };
    let dispatcher = match slot.as_ref() {
        Some(d) => d,
        None => return ERR_NOT_INITIALIZED,
    };

    let WgpuDispatcher {
        ref device,
        ref queue,
        ref pipelines,
        ref buffers,
        ..
    } = *dispatcher;

    let pipeline = match pipelines.get(&pipeline_id) {
        Some(p) => p,
        None => return ERR_NOT_FOUND,
    };

    let gpu_buf = match buffers.get(&buffer_id) {
        Some(b) => b,
        None => return ERR_NOT_FOUND,
    };

    let readback = match gpu_buf.readback.as_ref() {
        Some(r) => r,
        None => return ERR_INVALID_ARG,
    };

    if byte_len > gpu_buf.size {
        return ERR_INVALID_ARG;
    }

    // 1. Create a SINGLE CommandEncoder for the entire frame
    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("Axis Unified Frame Encoder"),
    });

    // 2. Compute Pass
    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("Axis Unified Compute Pass"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&pipeline.pipeline);
        
        let bind_group = match pipeline.bind_group.as_ref() {
            Some(bg) => bg,
            None => {
                readback.unmap();
                return ERR_INVALID_ARG;
            }
        };
        pass.set_bind_group(0, bind_group, &[]);
        pass.dispatch_workgroups(wg_x, wg_y, wg_z);
    }

    // 3. Copy to Readback Staging
    encoder.copy_buffer_to_buffer(&gpu_buf.buffer, 0, readback, 0, byte_len as wgpu::BufferAddress);

    // 4. Submit ONCE to the GPU
    queue.submit(Some(encoder.finish()));

    // 5. Map and Wait (Synchronize)
    use std::sync::mpsc;
    let (tx, rx) = mpsc::channel();
    readback.slice(..byte_len as u64).map_async(wgpu::MapMode::Read, move |r| {
        let _ = tx.send(r);
    });

    if device.poll(wgpu::PollType::wait_indefinitely()).is_err() {
        return ERR_POLL_FAILED;
    }
    if rx.recv().map_err(|_| ()).and_then(|r| r.map_err(|_| ())).is_err() {
        return ERR_MAP_FAILED;
    }

    // 6. Native Broadcast (Zero Julia Allocation)
    let broadcast_fn = {
        let guard = BROADCAST_CALLBACK.lock().unwrap();
        match *guard {
            Some(f) => f,
            None => {
                readback.unmap();
                return ERR_INVALID_ARG;
            }
        }
    };

    {
        let view = readback.slice(..byte_len as u64).get_mapped_range();
        
        let mut frame = Vec::with_capacity(17 + byte_len);
        frame.push(1);
        frame.extend_from_slice(&1u16.to_le_bytes());
        frame.extend_from_slice(&0u16.to_le_bytes());
        let ts = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos() as u64;
        frame.extend_from_slice(&ts.to_le_bytes());
        frame.extend_from_slice(&(byte_len as u32).to_le_bytes());
        frame.extend_from_slice(&view);

        unsafe {
            let path_bytes = path.as_bytes();
            broadcast_fn(
                path_bytes.as_ptr(),
                path_bytes.len(),
                frame.as_ptr(),
                frame.len(),
            );
        }
    }
    readback.unmap();

    OK
}

/// Set the Fomalhaut broadcast callback pointer
pub fn set_broadcast_callback(callback_ptr: *const std::ffi::c_void) -> i32 {
    if callback_ptr.is_null() {
        return ERR_NULL_PTR;
    }
    let mut guard = match BROADCAST_CALLBACK.lock() {
        Ok(g) => g,
        Err(_) => return ERR_PANIC,
    };
    unsafe {
        let func: WsBroadcastFn = std::mem::transmute(callback_ptr);
        *guard = Some(func);
    }
    OK
}


/// Create a compute pipeline from a WGSL source string.
///
/// `binding_flags_ptr` is a C array of `binding_count` u32 values, one per
/// @binding in the shader ( in binding-index order ).
/// Values: 0=storage_read, 1=storage_read_write, 2=uniform
pub fn create_compute_pipeline(
    id: usize,
    wgsl_ptr: *const std::os::raw::c_char,
    entry_ptr: *const std::os::raw::c_char,
    binding_flags_ptr: *const u32,
    binding_count: usize,
) -> i32 {
    if wgsl_ptr.is_null() || entry_ptr.is_null() || binding_flags_ptr.is_null() {
        return ERR_NULL_PTR;
    }

    let wgsl = unsafe {
        match std::ffi::CStr::from_ptr(wgsl_ptr).to_str() {
            Ok(s) => s.to_owned(),
            Err(_) => return ERR_INVALID_ARG,
        }
    };
    let entry = unsafe {
        match std::ffi::CStr::from_ptr(entry_ptr).to_str() {
            Ok(s) => s.to_owned(),
            Err(_) => return ERR_INVALID_ARG,
        }
    };
    let binding_flags =
        unsafe { std::slice::from_raw_parts(binding_flags_ptr, binding_count) };

    let mut slot = match dispatcher_slot().lock() {
        Ok(s) => s,
        Err(_) => return ERR_NOT_INITIALIZED,
    };
    let dispatcher = match slot.as_mut() {
        Some(d) => d,
        None => return ERR_NOT_INITIALIZED,
    };

    if dispatcher.pipelines.contains_key(&id) {
        return ERR_ALREADY_EXISTS;
    }

    // Build BindGroupLayout entries from the flag array
    let layout_entries: Vec<wgpu::BindGroupLayoutEntry> = binding_flags
        .iter()
        .enumerate()
        .map(|(i, &flag)| {
            let binding = i as u32;
            let ty = match flag {
                BINDING_STORAGE_READ => wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: true },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                BINDING_STORAGE_READ_WRITE => wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: false },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                BINDING_UNIFORM | _ => wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
            };
            wgpu::BindGroupLayoutEntry {
                binding,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty,
                count: None,
            }
        })
        .collect();

    let bind_group_layout =
        dispatcher
            .device
            .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some(&format!("Axis BGL {id}")),
                entries: &layout_entries,
            });

    let pipeline_layout =
        dispatcher
            .device
            .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some(&format!("Axis Pipeline Layout {id}")),
                bind_group_layouts: &[Some(&bind_group_layout)],
                immediate_size: 0,
            });

    let shader = dispatcher
        .device
        .create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some(&format!("Axis Shader {id}")),
            source: wgpu::ShaderSource::Wgsl(wgsl.into()),
        });

    let pipeline =
        dispatcher
            .device
            .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some(&format!("Axis Compute Pipeline {id}")),
                layout: Some(&pipeline_layout),
                module: &shader,
                entry_point: Some(&entry),
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                cache: None,
            });

    dispatcher.pipelines.insert(
        id,
        GpuPipeline {
            pipeline,
            bind_group_layout,
            bindings: Vec::new(),
            bind_group: None,
        },
    );

    OK
}

/// Bind a list of GPU buffers to a pipeline ( in binding-index order ).
/// Must be called after create_compute_pipeline and create_buffer.
pub fn bind_buffers(
    pipeline_id: usize,
    buffer_ids_ptr: *const usize,
    buffer_count: usize,
) -> i32 {
    if buffer_ids_ptr.is_null() {
        return ERR_NULL_PTR;
    }
    let buffer_ids =
        unsafe { std::slice::from_raw_parts(buffer_ids_ptr, buffer_count) };

    let mut slot = match dispatcher_slot().lock() {
        Ok(s) => s,
        Err(_) => return ERR_NOT_INITIALIZED,
    };
    let dispatcher = match slot.as_mut() {
        Some(d) => d,
        None => return ERR_NOT_INITIALIZED,
    };

    // Destructure to obtain disjoint field borrows:
    // `buffers` immutable + `pipelines` mutable + `device` immutable — all at once.
    let WgpuDispatcher {
        ref device,
        ref buffers,
        ref mut pipelines,
        ..
    } = *dispatcher;

    let pipeline = match pipelines.get_mut(&pipeline_id) {
        Some(p) => p,
        None => return ERR_NOT_FOUND,
    };

    // Validate all buffer IDs and build BindGroupEntries.
    // Entries borrow from `buffers`; `pipeline` borrows from `pipelines`.
    // Both are disjoint fields so the borrow checker is satisfied.
    let mut entries: Vec<wgpu::BindGroupEntry> = Vec::with_capacity(buffer_count);
    for (binding_index, &buf_id) in buffer_ids.iter().enumerate() {
        let gpu_buf = match buffers.get(&buf_id) {
            Some(b) => b,
            None => return ERR_NOT_FOUND,
        };
        entries.push(wgpu::BindGroupEntry {
            binding: binding_index as u32,
            resource: gpu_buf.buffer.as_entire_binding(),
        });
    }

    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some(&format!("Axis BindGroup {pipeline_id}")),
        layout: &pipeline.bind_group_layout,
        entries: &entries,
    });

    pipeline.bindings = buffer_ids
        .iter()
        .enumerate()
        .map(|(i, &id)| (id, i as u32))
        .collect();
    pipeline.bind_group = Some(bind_group);

    OK
}

/// Dispatch a compute pipeline. workgroup counts in x, y, z.
pub fn dispatch(pipeline_id: usize, wg_x: u32, wg_y: u32, wg_z: u32) -> i32 {
    let slot = match dispatcher_slot().lock() {
        Ok(s) => s,
        Err(_) => return ERR_NOT_INITIALIZED,
    };
    let dispatcher = match slot.as_ref() {
        Some(d) => d,
        None => return ERR_NOT_INITIALIZED,
    };

    let pipeline = match dispatcher.pipelines.get(&pipeline_id) {
        Some(p) => p,
        None => return ERR_NOT_FOUND,
    };

    let bind_group = match pipeline.bind_group.as_ref() {
        Some(bg) => bg,
        None => return ERR_INVALID_ARG, // bind_buffers not yet called
    };

    let mut encoder = dispatcher
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some(&format!("Axis Dispatch Encoder {pipeline_id}")),
        });

    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some(&format!("Axis Compute Pass {pipeline_id}")),
            timestamp_writes: None,
        });
        pass.set_pipeline(&pipeline.pipeline);
        pass.set_bind_group(0, bind_group, &[]);
        pass.dispatch_workgroups(wg_x, wg_y, wg_z);
    }

    dispatcher.queue.submit(Some(encoder.finish()));
    OK
}

/// Destroy a buffer ( free GPU memory ).
pub fn destroy_buffer(id: usize) -> i32 {
    let mut slot = match dispatcher_slot().lock() {
        Ok(s) => s,
        Err(_) => return ERR_NOT_INITIALIZED,
    };
    let dispatcher = match slot.as_mut() {
        Some(d) => d,
        None => return ERR_NOT_INITIALIZED,
    };

    if dispatcher.buffers.remove(&id).is_none() {
        return ERR_NOT_FOUND;
    }
    OK
}

/// Destroy a pipeline ( free GPU resources ).
pub fn destroy_pipeline(id: usize) -> i32 {
    let mut slot = match dispatcher_slot().lock() {
        Ok(s) => s,
        Err(_) => return ERR_NOT_INITIALIZED,
    };
    let dispatcher = match slot.as_mut() {
        Some(d) => d,
        None => return ERR_NOT_INITIALIZED,
    };

    if dispatcher.pipelines.remove(&id).is_none() {
        return ERR_NOT_FOUND;
    }
    OK
}
