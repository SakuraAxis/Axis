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

## Project Dependencies Details

wgpu License : [https://github.com/gfx-rs/wgpu/blob/trunk/LICENSE.MIT](https://github.com/gfx-rs/wgpu/blob/trunk/LICENSE.MIT) and [another Apache-2.0 License](https://github.com/gfx-rs/wgpu/blob/trunk/LICENSE.APACHE)
