# Axis FFI

![GitHub last commit](https://img.shields.io/github/last-commit/zzztzzzt/SakuraEngine.jl.svg)
![GitHub repo size](https://img.shields.io/github/repo-size/zzztzzzt/SakuraEngine.jl.svg)

<br>

### Axis - 3D Physics Simulation FFI For Web / Unity. - Julia / Rust / C#.

IMPORTANT : This project is still in the development and testing stages, licensing terms may be updated in the future. Please don't do any commercial usage currently.

## Project Dependencies Guide

**[ for Dependencies Details please see the end of this README ]**

[![Julia](https://img.shields.io/badge/Julia-9558B2?style=for-the-badge&logo=julia&logoColor=white)](https://github.com/JuliaLang/julia)
[![cudarc](https://img.shields.io/badge/cudarc-F04D23?style=for-the-badge&logo=rust&logoColor=white)](https://github.com/chelsea0x3b/cudarc)

Axis uses cudarc to call NVIDIA GPUs for parallel computing. cudarc licensed under the MIT License & Apache-2.0 License.

## How To Use

### Step 1. Install CUDA Toolkit

install at NVDIA Website : [https://developer.nvidia.com/cuda/toolkit](https://developer.nvidia.com/cuda/toolkit)

### Step 2. Change cudarc version at `axis_rs/Cargo.toml`

For example, if you use CUDA Toolkit 13.1, edit version to below

```toml
[dependencies]
cudarc = { version = "xx.xx.xx", features = ["cuda-13010"] }
```

## Project Dependencies Details

cudarc License : [https://github.com/chelsea0x3b/cudarc/blob/main/LICENSE-MIT](https://github.com/chelsea0x3b/cudarc/blob/main/LICENSE-MIT) and [another Apache-2.0 License](https://github.com/chelsea0x3b/cudarc/blob/main/LICENSE-APACHE)
