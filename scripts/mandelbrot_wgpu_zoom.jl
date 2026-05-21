#= CMD : julia --project=. --color=yes scripts/mandelbrot_wgpu_zoom.jl =#

import Axis as AX

@AX.rust_code """
use std::io::{self, Write};
use std::fmt::Write as FmtWrite;

fn unpack_rgb(packed: u32) -> (u32, u32, u32) {
    if packed == 0xFFFFFFFF {
        (0, 0, 0)
    } else {
        ((packed >> 16) & 0xFF, (packed >> 8) & 0xFF, packed & 0xFF)
    }
}
"""

@AX.rust_fn function axis_print_mandelbrot_ascii(
    pixels::Ptr{UInt32},
    width::UInt32,
    height::UInt32,
    _max_iter::UInt32
)::Cvoid
    """
    if pixels.is_null() || width == 0 || height == 0 { return; }

    let pixels = unsafe {
        std::slice::from_raw_parts(pixels, (width as usize) * (height as usize))
    };

    let mut out = String::with_capacity(((width + 1) * ((height + 1) / 2) * 48 + 32) as usize);

    out.push_str("\\x1b[2J\\x1b[H");

    for y in (0..height as usize).step_by(2) {
        for x in 0..width as usize {
            let top = pixels[y * width as usize + x];
            let bottom = if y + 1 < height as usize {
                pixels[(y + 1) * width as usize + x]
            } else {
                0xFFFFFFFF
            };

            let (tr, tg, tb) = unpack_rgb(top);
            let (br, bg, bb) = unpack_rgb(bottom);

            let _ = write!(
                out,
                "\\x1b[38;2;{};{};{}m\\x1b[48;2;{};{};{}m▀",
                tr, tg, tb, br, bg, bb
            );
        }
        out.push_str("\\x1b[0m\\n");
    }

    let _ = io::stdout().write_all(out.as_bytes());
    let _ = io::stdout().flush();
    """
end

const WIDTH          = length(ARGS) >= 1 ? parse(Int, ARGS[1])     : 100
const HEIGHT         = length(ARGS) >= 2 ? parse(Int, ARGS[2])     : 80
const FRAMES         = length(ARGS) >= 3 ? parse(Int, ARGS[3])     : 280
const DELAY_SECONDS  = length(ARGS) >= 4 ? parse(Float64, ARGS[4]) : 0.035
const MAX_ITER       = length(ARGS) >= 5 ? parse(Int, ARGS[5])     : 320

const PARAM_BUFFER_ID = 10_001
const PIXEL_BUFFER_ID = 10_002
const PIPELINE_ID     = 20_001

const MANDELBROT_WGSL = raw"""
struct Params {
    values: array<f32>,
};

struct Pixels {
    values: array<u32>,
};

@group(0) @binding(0) var<storage, read>       params: Params;
@group(0) @binding(1) var<storage, read_write> pixels: Pixels;

fn hsv_to_rgb(h: f32, s: f32, v: f32) -> vec3<f32> {
    let c   = v * s;
    let h6  = h * 6.0;
    let x   = c * (1.0 - abs(h6 % 2.0 - 1.0));
    var rgb: vec3<f32>;
    if      (h6 < 1.0) { rgb = vec3(c, x, 0.0); }
    else if (h6 < 2.0) { rgb = vec3(x, c, 0.0); }
    else if (h6 < 3.0) { rgb = vec3(0.0, c, x); }
    else if (h6 < 4.0) { rgb = vec3(0.0, x, c); }
    else if (h6 < 5.0) { rgb = vec3(x, 0.0, c); }
    else               { rgb = vec3(c, 0.0, x); }
    return rgb + vec3(v - c);
}

@compute @workgroup_size(16, 16, 1)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let width    = u32(params.values[0]);
    let height   = u32(params.values[1]);
    let max_iter = u32(params.values[2]);
    let center_x = params.values[4];
    let center_y = params.values[5];
    let scale    = params.values[6];
    let aspect   = params.values[7];

    if (id.x >= width || id.y >= height) { return; }

    let fx = (f32(id.x) / f32(width)  - 0.5) * scale * aspect;
    let fy = (f32(id.y) / f32(height) - 0.5) * scale;
    let cx = center_x + fx;
    let cy = center_y + fy;

    var zx   = 0.0;
    var zy   = 0.0;
    var iter = 0u;

    loop {
        if (iter >= max_iter || zx * zx + zy * zy > 4.0) { break; }
        let xt = zx * zx - zy * zy + cx;
        zy = 2.0 * zx * zy + cy;
        zx = xt;
        iter = iter + 1u;
    }

    // Inside the set => Pure black, special marker
    if (iter >= max_iter) {
        pixels.values[id.y * width + id.x] = 0xFFFFFFFFu;
        return;
    }

    // Smooth coloring ( eliminate banding )
    let log_zn   = log(zx * zx + zy * zy) * 0.5;
    let nu       = log(log_zn / log(2.0)) / log(2.0);
    let smooth_n = f32(iter) + 1.0 - nu;

    // HSV Color Wheel Mapping
    let hue_speed = 0.08;   // Decrease the setting = longer period, slower speed; Increase the setting = shorter period, more colors
    let hue = fract(smooth_n * hue_speed);
    let sat = 0.85;         // Saturation : 0.0 grayscale ~ 1.0 most vivid
    let val = 1.0;          // brightness

    let rgb = hsv_to_rgb(hue, sat, val);
    let r   = u32(clamp(rgb.x, 0.0, 1.0) * 255.0);
    let g   = u32(clamp(rgb.y, 0.0, 1.0) * 255.0);
    let b   = u32(clamp(rgb.z, 0.0, 1.0) * 255.0);

    pixels.values[id.y * width + id.x] = (r << 16u) | (g << 8u) | b;
}
"""

function params_for_frame(frame::Int)
    center_x = Float32(-0.7436439)
    center_y = Float32(0.13182591)
    scale    = Float32(3.0 * 0.955^frame)
    aspect   = Float32(WIDTH / HEIGHT)

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
    WIDTH    > 0 || error("width must be positive")
    HEIGHT   > 0 || error("height must be positive")
    FRAMES   > 0 || error("frames must be positive")
    MAX_ITER > 0 || error("max_iter must be positive")

    AX.bridge_up(abspath(joinpath(@__DIR__, "..", "axis_rs")))

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
