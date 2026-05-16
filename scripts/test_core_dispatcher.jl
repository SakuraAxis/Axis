using Axis

#=
Core Dispatcher Smoke Test
Tests only the generic wgpu_* API - no ocean physics involved.
Shader : doubles every element of an input float32 array.
=#

const DOUBLE_WGSL = """
@group(0) @binding(0) var<storage, read> input : array<f32>;
@group(0) @binding(1) var<storage, read_write> output : array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
    let i = gid.x;
    if (i >= arrayLength(&input)) { return; }
    output[i] = input[i] * 2.0;
}
"""

function test_core_dispatcher()
    println("||||||  Axis Core Dispatcher Smoke Test  ||||||\n")

    println("Library path : ", Axis.axis_rs_library_path())
    println("Library found : ", Axis.axis_rs_available())
    Axis.axis_rs_available() || error("axis_rs release library not found.")

    # Resource IDs
    BUF_INPUT    = 101
    BUF_OUTPUT   = 102
    PIPELINE_ID  = 101
    N            = 16
    BYTE_SIZE    = N * sizeof(Float32)

    # Initialise WGPU dispatcher
    Axis.wgpu_init!()
    println("wgpu_init!() OK")

    # Create buffers
    Axis.wgpu_create_buffer!(BUF_INPUT,  BYTE_SIZE, Axis.BINDING_STORAGE_READ)
    Axis.wgpu_create_buffer!(BUF_OUTPUT, BYTE_SIZE, Axis.BINDING_STORAGE_READ_WRITE)
    println("Buffers created OK")

    # Upload input data: [1.0, 2.0, ..., N]
    input_data = Float32.(1:N)
    Axis.wgpu_write_buffer!(BUF_INPUT, input_data)
    println("Buffer write OK")

    # Compile shader and bind
    Axis.wgpu_create_compute_pipeline!(
        PIPELINE_ID, DOUBLE_WGSL, "main",
        UInt32[Axis.BINDING_STORAGE_READ, Axis.BINDING_STORAGE_READ_WRITE],
    )
    Axis.wgpu_bind_buffers!(PIPELINE_ID, [BUF_INPUT, BUF_OUTPUT])
    println("Pipeline compile & bind OK")

    # Dispatch
    wg_x = cld(N, 64)
    Axis.wgpu_dispatch!(PIPELINE_ID; wg_x = wg_x)
    println("Dispatch OK")

    # Readback and verify
    output_data = Vector{Float32}(undef, N)
    Axis.wgpu_read_buffer!(BUF_OUTPUT, output_data)
    println("Readback OK")

    expected = Float32.(2:2:(2 * N))
    all(output_data .== expected) ||
        error("Result mismatch!\n got : $output_data\n expected : $expected")

    println("Result verified: input * 2 = output")

    # Cleanup
    Axis.wgpu_destroy_pipeline!(PIPELINE_ID)
    Axis.wgpu_destroy_buffer!(BUF_INPUT)
    Axis.wgpu_destroy_buffer!(BUF_OUTPUT)
    println("\nCore dispatcher smoke test PASSED.")
end

test_core_dispatcher()
