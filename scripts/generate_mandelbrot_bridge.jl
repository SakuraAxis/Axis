#=
Generate Rust bridge code for scripts/mandelbrot_wgpu_zoom.jl.

Run from the workspace root:
  julia --project=. scripts/generate_mandelbrot_bridge.jl
=#

using Pkg
Pkg.activate(abspath(joinpath(@__DIR__, "..")))

@info "Loading Mandelbrot WGPU script declarations..."
include(joinpath(@__DIR__, "mandelbrot_wgpu_zoom.jl"))

import Axis as AX

axis_generated_dir = abspath(joinpath(@__DIR__, "..", "axis_rs", "src", "generated"))

@info "Triggering Axis Rust code generator..." axis_generated_dir
AX.generate_bridge(axis_generated_dir)

@info "Generation complete! Rust files successfully written to: $axis_generated_dir"
