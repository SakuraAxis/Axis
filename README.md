# Axis FFI

![GitHub last commit](https://img.shields.io/github/last-commit/zzztzzzt/SakuraEngine.jl.svg)
![GitHub repo size](https://img.shields.io/github/repo-size/zzztzzzt/SakuraEngine.jl.svg)

<br>

<img src="https://github.com/SakuraAxis/Axis/blob/main/logo/logo.png" alt="axis-logo" style="height: 280px; width: auto;" />

### Axis - Lightweight Julia-Rust FFI bridge + WGPU compute dispatcher.

IMPORTANT : This project is still in the development and testing stages, licensing terms may be updated in the future. Please don't do any commercial usage currently.

## Project Dependencies Guide

[![Julia](https://img.shields.io/badge/Julia-9558B2?style=for-the-badge&logo=julia&logoColor=white)](https://github.com/JuliaLang/julia)
[![wgpu](https://img.shields.io/badge/wgpu-F04D23?style=for-the-badge&logo=rust&logoColor=white)](https://github.com/gfx-rs/wgpu)

**[ for Dependencies Details please see the end of this README ]**

Axis uses wgpu to call GPU on Rust side for computing. wgpu licensed under the MIT License & Apache-2.0 License.

## Try Example Code

### Start Test Script ( Mandelbrot Zoom )

`julia`

`] activate .`

`instantiate`

`dev ./Axis`

and press backspace to close pkg mode

press Ctrl + D to close Julia REPL

`julia --project=. --color=yes scripts/mandelbrot_wgpu_zoom.jl`

run below to cancel ffi bridge :

`julia --project=. scripts/clean_bridge.jl`

## How To Use

### Basic Rules

```julia
import Axis as AX
```

if a julia file uses Axis, it will be compiled to a `mirror-rust-file`. for example, `mandelbrot_wgpu_zoom.jl` => `mandelbrot_wgpu_zoom.rs` ( OR `testABC.jl` => `test_abc.rs` ).

the `@AX.rust_code` & `@AX.rust_fn` in **Same Julia File** will be compiled into **Same Rust File**.

### Build / Cancel FFI Bridge

activate bridge :

```julia
AX.bridge_up(abspath(joinpath(@__DIR__, "to", "axis_rs")))
```

cancel bridge ( & clear generated files ) :

```julia
AX.bridge_down(abspath(joinpath(@__DIR__, "to", "axis_rs")))
```

### @AX.rust_code

**@AX.rust_code** let you write the `static` code to mirror-rust-file, which means this type of code won't be opened to Julia.

use this macro to import package, declare private function, set the file-scope variables...

```julia
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
```

### @AX.rust_fn + @AX.call_rust_fn

**@AX.rust_fn** & **@AX.call_rust_fn** let you declare Rust Foreign Function in Julia directly. and call it directly.

define Foreign Function :

```julia
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
                "\\x1b[38;2;{};{};{}m\\x1b[48;2;{};{};{}m\\u{2580}",
                tr, tg, tb, br, bg, bb
            );
        }
        out.push_str("\\x1b[0m\\n");
    }

    let _ = io::stdout().write_all(out.as_bytes());
    let _ = io::stdout().flush();
    """
end
```

call Foreign Function :

```julia
@AX.call_rust_fn axis_print_mandelbrot_ascii(
    pointer(pixels),
    UInt32(WIDTH),
    UInt32(HEIGHT),
    UInt32(MAX_ITER),
)
```

### AX.wgpu_init!

Initialise the WGPU device and compute dispatcher. Safe to call multiple times.

```julia
AX.wgpu_init!()
```

### AX.wgpu_create_buffer!

Create a GPU buffer with a user-defined integer ID, byte size, and binding type.

```julia
AX.wgpu_create_buffer!(1, sizeof(input), AX.BINDING_STORAGE_READ)
AX.wgpu_create_buffer!(2, sizeof(output), AX.BINDING_STORAGE_READ_WRITE)
```

binding types :

- `AX.BINDING_STORAGE_READ` : read-only storage buffer
- `AX.BINDING_STORAGE_READ_WRITE` : read-write storage buffer, supports readback
- `AX.BINDING_UNIFORM` : uniform buffer

### AX.wgpu_write_buffer!

Write a Julia `Array` or `Vector` into an existing GPU buffer.

```julia
AX.wgpu_write_buffer!(1, input)
```

### AX.wgpu_read_buffer!

Read data from a GPU buffer into a pre-allocated Julia `Array` or `Vector`.

```julia
AX.wgpu_read_buffer!(2, output)
```

### AX.wgpu_destroy_buffer!

Destroy a GPU buffer and release its GPU-side resources.

```julia
AX.wgpu_destroy_buffer!(1)
```

### AX.wgpu_create_compute_pipeline!

Compile WGSL source and register it as a compute pipeline with a user-defined integer ID.

```julia
AX.wgpu_create_compute_pipeline!(
    100,
    WGSL_SOURCE,
    "main",
    UInt32[AX.BINDING_STORAGE_READ, AX.BINDING_STORAGE_READ_WRITE],
)
```

### AX.wgpu_bind_buffers!

Bind buffers to a pipeline. The buffer order must match WGSL `@binding(n)` order.

```julia
AX.wgpu_bind_buffers!(100, [1, 2])
```

### AX.wgpu_dispatch!

Run the compute pipeline with the selected workgroup counts.

```julia
AX.wgpu_dispatch!(100; wg_x=cld(length(output), 64))
```

### AX.wgpu_destroy_pipeline!

Destroy a compute pipeline and release its GPU-side resources.

```julia
AX.wgpu_destroy_pipeline!(100)
```

notes :

- buffer and pipeline IDs are user-defined integers, keep them unique.
- call `AX.bridge_up(...)` before using WGPU APIs when the Rust bridge needs to be generated or rebuilt.
- use `try ... finally` for cleanup in longer scripts.
- see `scripts/mandelbrot_wgpu_zoom.jl` for a complete runnable example.

## Project Dependencies Details

wgpu License : [https://github.com/gfx-rs/wgpu/blob/trunk/LICENSE.MIT](https://github.com/gfx-rs/wgpu/blob/trunk/LICENSE.MIT) and [another Apache-2.0 License](https://github.com/gfx-rs/wgpu/blob/trunk/LICENSE.APACHE)
