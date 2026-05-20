using Pkg

const ROOT = normpath(joinpath(@__DIR__, ".."))
Pkg.activate(ROOT)

import Axis as AX

@AX.rust_fn function axis_print_mandelbrot_ascii(
    pixels::Ptr{UInt32},
    width::UInt32,
    height::UInt32,
    max_iter::UInt32
)::Cvoid
    """
    use std::io::{self, Write};

    if pixels.is_null() || width == 0 || height == 0 {
        return;
    }

    let pixels = unsafe { std::slice::from_raw_parts(pixels, (width as usize) * (height as usize)) };
    let ramp = b" .:-=+*#%@";
    let mut out = String::with_capacity(((width + 1) * height + 16) as usize);

    out.push_str("\\x1b[2J\\x1b[H");
    for y in 0..height as usize {
        for x in 0..width as usize {
            let n = pixels[y * width as usize + x];
            let ch = if n >= max_iter {
                b' '
            } else {
                let idx = ((n as usize) * (ramp.len() - 1)) / (max_iter.max(1) as usize);
                ramp[idx]
            };
            out.push(ch as char);
        }
        out.push('\\n');
    }

    let _ = io::stdout().write_all(out.as_bytes());
    let _ = io::stdout().flush();
    """
end

const WIDTH = length(ARGS) >= 1 ? parse(Int, ARGS[1]) : 100
const HEIGHT = length(ARGS) >= 2 ? parse(Int, ARGS[2]) : 40
const FRAMES = length(ARGS) >= 3 ? parse(Int, ARGS[3]) : 160
const DELAY_SECONDS = length(ARGS) >= 4 ? parse(Float64, ARGS[4]) : 0.035
const MAX_ITER = length(ARGS) >= 5 ? parse(Int, ARGS[5]) : 240

const PARAM_BUFFER_ID = 10_001
const PIXEL_BUFFER_ID = 10_002
const PIPELINE_ID = 20_001

const MANDELBROT_WGSL = raw"""
struct Params {
    values: array<f32>,
};

struct Pixels {
    values: array<u32>,
};

@group(0) @binding(0) var<storage, read> params: Params;
@group(0) @binding(1) var<storage, read_write> pixels: Pixels;

@compute @workgroup_size(16, 16, 1)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let width = u32(params.values[0]);
    let height = u32(params.values[1]);
    let max_iter = u32(params.values[2]);
    let center_x = params.values[4];
    let center_y = params.values[5];
    let scale = params.values[6];
    let aspect = params.values[7];

    if (id.x >= width || id.y >= height) {
        return;
    }

    let fx = (f32(id.x) / f32(width) - 0.5) * scale * aspect;
    let fy = (f32(id.y) / f32(height) - 0.5) * scale;
    let cx = center_x + fx;
    let cy = center_y + fy;

    var zx = 0.0;
    var zy = 0.0;
    var iter = 0u;

    loop {
        if (iter >= max_iter || zx * zx + zy * zy > 4.0) {
            break;
        }

        let xt = zx * zx - zy * zy + cx;
        zy = 2.0 * zx * zy + cy;
        zx = xt;
        iter = iter + 1u;
    }

    pixels.values[id.y * width + id.x] = iter;
}
"""

function params_for_frame(frame::Int)
    center_x = Float32(-0.7436439)
    center_y = Float32(0.13182591)
    scale = Float32(3.0 * 0.955^frame)
    aspect = Float32(WIDTH / HEIGHT)

    return Float32[
        Float32(WIDTH),
        Float32(HEIGHT),
        Float32(MAX_ITER),
        0.0f0,
        center_x,
        center_y,
        scale,
        aspect,
    ]
end

function main()
    WIDTH > 0 || error("width must be positive")
    HEIGHT > 0 || error("height must be positive")
    FRAMES > 0 || error("frames must be positive")
    MAX_ITER > 0 || error("max_iter must be positive")

    AX.axis_rs_available() ||
        error("Axis Rust library is not available at $(AX.axis_rs_library_path()). Run scripts/generate_mandelbrot_bridge.jl, then build axis_rs with `cargo build --release`.")

    AX.wgpu_init!()

    pixels = Vector{UInt32}(undef, WIDTH * HEIGHT)
    params = params_for_frame(0)

    AX.wgpu_create_buffer!(PARAM_BUFFER_ID, sizeof(params), AX.BINDING_STORAGE_READ)
    AX.wgpu_create_buffer!(PIXEL_BUFFER_ID, sizeof(pixels), AX.BINDING_STORAGE_READ_WRITE)
    AX.wgpu_create_compute_pipeline!(
        PIPELINE_ID,
        MANDELBROT_WGSL,
        "main",
        UInt32[AX.BINDING_STORAGE_READ, AX.BINDING_STORAGE_READ_WRITE],
    )
    AX.wgpu_bind_buffers!(PIPELINE_ID, [PARAM_BUFFER_ID, PIXEL_BUFFER_ID])

    try
        for frame in 0:(FRAMES - 1)
            params .= params_for_frame(frame)
            AX.wgpu_write_buffer!(PARAM_BUFFER_ID, params)
            AX.wgpu_dispatch!(PIPELINE_ID; wg_x=cld(WIDTH, 16), wg_y=cld(HEIGHT, 16))
            AX.wgpu_read_buffer!(PIXEL_BUFFER_ID, pixels)

            @AX.call_rust_fn axis_print_mandelbrot_ascii(
                pointer(pixels),
                UInt32(WIDTH),
                UInt32(HEIGHT),
                UInt32(MAX_ITER),
            )

            println("frame $(frame + 1)/$FRAMES  scale=$(round(Float64(params[7]); sigdigits=5))")
            sleep(DELAY_SECONDS)
        end
    finally
        AX.wgpu_destroy_pipeline!(PIPELINE_ID)
        AX.wgpu_destroy_buffer!(PIXEL_BUFFER_ID)
        AX.wgpu_destroy_buffer!(PARAM_BUFFER_ID)
    end

    return nothing
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
