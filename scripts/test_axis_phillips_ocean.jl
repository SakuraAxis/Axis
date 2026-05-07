using Axis

function test_axis_phillips_ocean()
    println("Axis Rust library: ", Axis.axis_rs_library_path())
    println("Axis Rust library available: ", Axis.axis_rs_available())

    Axis.axis_rs_available() || error("axis_rs release library was not found.")

    windx, windy = Axis.normalize2(0.92f0, 0.38f0)
    value = Axis.phillips_spectrum(1.0f0, 0.2f0, windx, windy)
    zero_value = Axis.phillips_spectrum(0.0f0, 0.0f0, windx, windy)
    opposite_value = Axis.phillips_spectrum(-1.0f0, -0.2f0, windx, windy)

    println("normalized wind: (", windx, ", ", windy, ")")
    println("phillips_spectrum(1.0, 0.2): ", value)
    println("phillips_spectrum(0.0, 0.0): ", zero_value)
    println("phillips_spectrum(-1.0, -0.2): ", opposite_value)

    isfinite(value) || error("sample spectrum value is not finite.")
    value > 0.0f0 || error("sample spectrum value should be positive.")
    zero_value == 0.0f0 || error("zero wave vector should return 0.")
    opposite_value == 0.0f0 || error("opposite wind direction should return 0.")

    println("Axis Phillips ocean smoke test passed.")
    return value
end

test_axis_phillips_ocean()
