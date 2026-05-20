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

## How To Use

### Start Test Script ( Mandelbrot Zoom )

`julia --project=. scripts/generate_mandelbrot_bridge.jl`

`cd axis_rs`

`cargo build --release`

`cd ..`

`julia --project=. scripts/mandelbrot_wgpu_zoom.jl`

## Project Dependencies Details

wgpu License : [https://github.com/gfx-rs/wgpu/blob/trunk/LICENSE.MIT](https://github.com/gfx-rs/wgpu/blob/trunk/LICENSE.MIT) and [another Apache-2.0 License](https://github.com/gfx-rs/wgpu/blob/trunk/LICENSE.APACHE)
