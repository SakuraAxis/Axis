use std::{
    future::Future,
    pin::Pin,
    slice,
    sync::{Mutex, OnceLock, mpsc},
    task::{Context, Poll, Wake, Waker},
    thread,
};

use wgpu::util::DeviceExt;

const WORKGROUP_SIZE: u32 = 256;

const SHADER: &str = r#"
struct Params {
    frame_count: u32,
    component_count: u32,
    time: f32,
    _pad: f32,
}

@group(0) @binding(0) var<storage, read_write> frame: array<f32>;
@group(0) @binding(1) var<storage, read> phase_base: array<f32>;
@group(0) @binding(2) var<storage, read> components: array<vec4<f32>>;
@group(0) @binding(3) var<uniform> params: Params;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    if (idx >= params.frame_count) {
        return;
    }

    var height = 0.0;
    for (var component = 0u; component < params.component_count; component = component + 1u) {
        let phase_index = idx + component * params.frame_count;
        let data = components[component];
        height = height + data.y *
            cos(phase_base[phase_index] - data.x * params.time + data.z);
    }

    frame[idx] = height;
}
"#;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WgpuError {
    AlreadyInitialized,
    NoAdapter,
    RequestDeviceFailed,
    NotInitialized,
    InvalidFrameCount,
    InvalidComponentCount,
    BufferTooSmall,
    MapFailed,
    PollFailed,
    ReadbackFailed,
    BuffersNotUploaded,
    FrameCountMismatch,
}

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

fn block_on<F: Future>(future: F) -> F::Output {
    let waker = Waker::from(std::sync::Arc::new(ThreadWaker {
        thread: thread::current(),
    }));
    let mut context = Context::from_waker(&waker);
    let mut future = std::pin::pin!(future);

    loop {
        match Future::poll(Pin::as_mut(&mut future), &mut context) {
            Poll::Ready(value) => return value,
            Poll::Pending => thread::park(),
        }
    }
}

/// GPU resources that persist across frames.
struct PersistentBuffers {
    frame_buffer: wgpu::Buffer,
    readback_buffer: wgpu::Buffer,
    params_buffer: wgpu::Buffer,
    bind_group: wgpu::BindGroup,
    frame_count: usize,
    component_count: usize,
}

struct WgpuBackend {
    device: wgpu::Device,
    queue: wgpu::Queue,
    pipeline: wgpu::ComputePipeline,
    bind_group_layout: wgpu::BindGroupLayout,
    persistent: Option<PersistentBuffers>,
}

static BACKEND: OnceLock<Mutex<Option<WgpuBackend>>> = OnceLock::new();

fn backend_slot() -> &'static Mutex<Option<WgpuBackend>> {
    BACKEND.get_or_init(|| Mutex::new(None))
}

pub fn init() -> Result<(), WgpuError> {
    let mut slot = backend_slot()
        .lock()
        .map_err(|_| WgpuError::RequestDeviceFailed)?;
    if slot.is_some() {
        return Ok(());
    }

    let instance = wgpu::Instance::default();
    let adapter = block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
        power_preference: wgpu::PowerPreference::HighPerformance,
        force_fallback_adapter: false,
        compatible_surface: None,
    }))
    .map_err(|_| WgpuError::NoAdapter)?;

    let (device, queue) = block_on(adapter.request_device(&wgpu::DeviceDescriptor {
        label: Some("Axis Phillips Ocean Device"),
        required_features: wgpu::Features::empty(),
        required_limits: wgpu::Limits::downlevel_defaults(),
        experimental_features: wgpu::ExperimentalFeatures::disabled(),
        memory_hints: wgpu::MemoryHints::Performance,
        trace: wgpu::Trace::Off,
    }))
    .map_err(|_| WgpuError::RequestDeviceFailed)?;

    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("Axis Phillips Ocean Compute Shader"),
        source: wgpu::ShaderSource::Wgsl(SHADER.into()),
    });

    let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("Axis Phillips Ocean Bind Group Layout"),
        entries: &[
            storage_entry(0, false),
            storage_entry(1, true),
            storage_entry(2, true),
            uniform_entry(3),
        ],
    });

    let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("Axis Phillips Ocean Pipeline Layout"),
        bind_group_layouts: &[Some(&bind_group_layout)],
        immediate_size: 0,
    });

    let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("Axis Phillips Ocean Compute Pipeline"),
        layout: Some(&pipeline_layout),
        module: &shader,
        entry_point: Some("main"),
        compilation_options: wgpu::PipelineCompilationOptions::default(),
        cache: None,
    });

    *slot = Some(WgpuBackend {
        device,
        queue,
        pipeline,
        bind_group_layout,
        persistent: None,
    });

    Ok(())
}

/// Upload persistent buffers to the GPU. Call once after init().
pub fn upload_buffers(
    phase_base: &[f32],
    omega: &[f32],
    amp: &[f32],
    phase0: &[f32],
    frame_count: usize,
    component_count: usize,
) -> Result<(), WgpuError> {
    if frame_count == 0 {
        return Err(WgpuError::InvalidFrameCount);
    }
    if component_count == 0 {
        return Err(WgpuError::InvalidComponentCount);
    }

    let phase_base_len = frame_count
        .checked_mul(component_count)
        .ok_or(WgpuError::BufferTooSmall)?;

    if phase_base.len() < phase_base_len
        || omega.len() < component_count
        || amp.len() < component_count
        || phase0.len() < component_count
    {
        return Err(WgpuError::BufferTooSmall);
    }

    let mut slot = backend_slot()
        .lock()
        .map_err(|_| WgpuError::NotInitialized)?;
    let backend = slot.as_mut().ok_or(WgpuError::NotInitialized)?;

    backend.upload_buffers(
        &phase_base[..phase_base_len],
        &omega[..component_count],
        &amp[..component_count],
        &phase0[..component_count],
        frame_count,
        component_count,
    );

    Ok(())
}

/// Compute a wave frame. Only updates the time parameter and dispatches.
pub fn compute_wave(
    frame: &mut [f32],
    frame_count: usize,
    time: f32,
) -> Result<(), WgpuError> {
    if frame_count == 0 {
        return Err(WgpuError::InvalidFrameCount);
    }
    if frame.len() < frame_count {
        return Err(WgpuError::BufferTooSmall);
    }

    let slot = backend_slot()
        .lock()
        .map_err(|_| WgpuError::NotInitialized)?;
    let backend = slot.as_ref().ok_or(WgpuError::NotInitialized)?;
    backend.compute_wave(&mut frame[..frame_count], frame_count, time)
}

fn storage_entry(binding: u32, read_only: bool) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Storage { read_only },
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}

fn uniform_entry(binding: u32) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Uniform,
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}

fn f32_slice_as_bytes(data: &[f32]) -> &[u8] {
    unsafe { slice::from_raw_parts(data.as_ptr().cast::<u8>(), std::mem::size_of_val(data)) }
}

fn params_bytes(frame_count: usize, component_count: usize, time: f32) -> [u8; 16] {
    let mut bytes = [0u8; 16];
    bytes[0..4].copy_from_slice(&(frame_count as u32).to_ne_bytes());
    bytes[4..8].copy_from_slice(&(component_count as u32).to_ne_bytes());
    bytes[8..12].copy_from_slice(&time.to_ne_bytes());
    bytes
}

fn component_bytes(omega: &[f32], amp: &[f32], phase0: &[f32]) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(omega.len() * 16);
    for ((omega, amp), phase0) in omega.iter().zip(amp).zip(phase0) {
        bytes.extend_from_slice(&omega.to_ne_bytes());
        bytes.extend_from_slice(&amp.to_ne_bytes());
        bytes.extend_from_slice(&phase0.to_ne_bytes());
        bytes.extend_from_slice(&0.0f32.to_ne_bytes());
    }
    bytes
}

impl WgpuBackend {
    fn upload_buffers(
        &mut self,
        phase_base: &[f32],
        omega: &[f32],
        amp: &[f32],
        phase0: &[f32],
        frame_count: usize,
        component_count: usize,
    ) {
        let frame_bytes = (frame_count * std::mem::size_of::<f32>()) as wgpu::BufferAddress;

        let frame_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Axis Phillips Ocean Frame Buffer"),
            size: frame_bytes,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let phase_base_buffer = self.create_storage_buffer("Phase Base", phase_base);

        let component_data = component_bytes(omega, amp, phase0);
        let components_buffer =
            self.create_bytes_storage_buffer("Phillips Ocean Components", &component_data);

        let params_buffer = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Axis Phillips Ocean Params Buffer"),
                contents: &params_bytes(frame_count, component_count, 0.0),
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            });

        let readback_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Axis Phillips Ocean Readback Buffer"),
            size: frame_bytes,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });

        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Axis Phillips Ocean Bind Group"),
            layout: &self.bind_group_layout,
            entries: &[
                bind_entry(0, &frame_buffer),
                bind_entry(1, &phase_base_buffer),
                bind_entry(2, &components_buffer),
                bind_entry(3, &params_buffer),
            ],
        });

        self.persistent = Some(PersistentBuffers {
            frame_buffer,
            readback_buffer,
            params_buffer,
            bind_group,
            frame_count,
            component_count,
        });
    }

    fn compute_wave(
        &self,
        frame: &mut [f32],
        frame_count: usize,
        time: f32,
    ) -> Result<(), WgpuError> {
        let persistent = self.persistent.as_ref().ok_or(WgpuError::BuffersNotUploaded)?;

        if frame_count != persistent.frame_count {
            return Err(WgpuError::FrameCountMismatch);
        }

        // Update only the params uniform (16 bytes) — the only thing that changes per frame
        self.queue.write_buffer(
            &persistent.params_buffer,
            0,
            &params_bytes(persistent.frame_count, persistent.component_count, time),
        );

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("Axis Phillips Ocean Compute Encoder"),
            });

        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Axis Phillips Ocean Compute Pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &persistent.bind_group, &[]);
            pass.dispatch_workgroups((frame_count as u32).div_ceil(WORKGROUP_SIZE), 1, 1);
        }

        encoder.copy_buffer_to_buffer(
            &persistent.frame_buffer,
            0,
            &persistent.readback_buffer,
            0,
            (frame_count * std::mem::size_of::<f32>()) as wgpu::BufferAddress,
        );
        self.queue.submit(Some(encoder.finish()));

        let (sender, receiver) = mpsc::channel();
        persistent
            .readback_buffer
            .slice(..)
            .map_async(wgpu::MapMode::Read, move |result| {
                let _ = sender.send(result);
            });

        self.device
            .poll(wgpu::PollType::wait_indefinitely())
            .map_err(|_| WgpuError::PollFailed)?;

        receiver
            .recv()
            .map_err(|_| WgpuError::ReadbackFailed)?
            .map_err(|_| WgpuError::MapFailed)?;

        {
            let view = persistent.readback_buffer.slice(..).get_mapped_range();
            for (target, bytes) in frame.iter_mut().zip(view.chunks_exact(4)) {
                *target = f32::from_ne_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]);
            }
        }
        persistent.readback_buffer.unmap();

        Ok(())
    }

    fn create_storage_buffer(&self, label: &str, data: &[f32]) -> wgpu::Buffer {
        self.create_bytes_storage_buffer(label, f32_slice_as_bytes(data))
    }

    fn create_bytes_storage_buffer(&self, label: &str, data: &[u8]) -> wgpu::Buffer {
        self.device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some(label),
                contents: data,
                usage: wgpu::BufferUsages::STORAGE,
            })
    }
}

fn bind_entry(binding: u32, buffer: &wgpu::Buffer) -> wgpu::BindGroupEntry<'_> {
    wgpu::BindGroupEntry {
        binding,
        resource: buffer.as_entire_binding(),
    }
}
