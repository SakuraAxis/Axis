using Axis
using Axis.Ocean

function test_axis_phillips_ocean()
    println("||||||  Axis Phillips Ocean Smoke Test  ||||||\n")

    println("Library path : ", Axis.axis_rs_library_path())
    println("Library found: ", Axis.axis_rs_available())
    Axis.axis_rs_available() || error("axis_rs release library not found.")

    # Spectrum math ( pure Julia, no GPU required )
    windx, windy = Axis.normalize2(0.92f0, 0.38f0)
    value         = Axis.Ocean.phillips_spectrum(1.0f0,  0.2f0, windx, windy)
    zero_value    = Axis.Ocean.phillips_spectrum(0.0f0,  0.0f0, windx, windy)
    opposite      = Axis.Ocean.phillips_spectrum(-1.0f0, -0.2f0, windx, windy)

    println("Normalized wind  : ($windx, $windy)")
    println("Spectrum(1,0.2)  : $value")
    println("Spectrum(0,0)    : $zero_value   (expected 0)")
    println("Spectrum(-1,-0.2): $opposite  (expected 0)\n")

    isfinite(value)    || error("Spectrum value is not finite.")
    value > 0f0        || error("Spectrum value should be positive.")
    zero_value == 0f0  || error("Zero wave vector should return 0.")
    opposite == 0f0    || error("Opposite wind direction should return 0.")

    # Full GPU simulation
    sim = Axis.Ocean.create_phillips_sim(
        resolution      = 96,
        component_count = 128,
        wind_speed      = 14.0,
        wind_direction  = (0.92, 0.38),
        seed            = 42,
    )
    Axis.Ocean.init!(sim)

    frame = Axis.Ocean.compute_wave!(sim, 0.0)

    println("First component kx    : ", sim.kx[1])
    println("First component omega : ", sim.omega[1])
    println("First component amp   : ", sim.amp[1])
    println("First frame height[1] : ", frame[1])

    all(isfinite, sim.kx)    || error("kx contains non-finite values.")
    all(isfinite, sim.omega) || error("omega contains non-finite values.")
    all(isfinite, sim.amp)   || error("amp contains non-finite values.")
    all(isfinite, frame)     || error("frame_buffer contains non-finite values.")

    # Test multiple simultaneous instances
    sim2 = Axis.Ocean.create_phillips_sim(wind_speed = 7.0, seed = 99)
    Axis.Ocean.init!(sim2)
    frame2 = Axis.Ocean.compute_wave!(sim2, 0.0)
    all(isfinite, frame2) || error("sim2 frame_buffer contains non-finite values.")
    println("\nMultiple instances OK (sim1 vs sim2 differ: $(frame[1] != frame2[1]))")

    # Cleanup
    Axis.Ocean.destroy!(sim)
    Axis.Ocean.destroy!(sim2)
    println("\nAxis Phillips Ocean smoke test PASSED.")

    return frame
end

test_axis_phillips_ocean()
