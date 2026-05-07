pub fn phillips_spectrum(
    kx: f32,
    ky: f32,
    wind_x: f32,
    wind_y: f32,
    wind_speed: f32,
    gravity: f32,
) -> f32 {
    let k2 = kx * kx + ky * ky;
    if k2 < 1e-6 {
        return 0.0;
    }

    let k = k2.sqrt();
    let alignment = ((kx / k) * wind_x + (ky / k) * wind_y).max(0.0);

    let l = (wind_speed * wind_speed) / gravity;
    let l2_small = (l * 0.0015).powi(2);

    (-1.0 / (k2 * l * l)).exp() / (k2 * k2) * alignment.powi(4) * (-k2 * l2_small).exp()
}

#[cfg(test)]
mod tests {
    use super::phillips_spectrum;

    #[test]
    fn returns_zero_for_near_zero_wave_vector() {
        assert_eq!(phillips_spectrum(0.0, 0.0, 1.0, 0.0, 14.0, 9.81), 0.0);
    }

    #[test]
    fn returns_zero_for_opposite_wind_direction() {
        assert_eq!(phillips_spectrum(-1.0, 0.0, 1.0, 0.0, 14.0, 9.81), 0.0);
    }

    #[test]
    fn returns_positive_value_for_aligned_wave() {
        let value = phillips_spectrum(1.0, 0.2, 0.9242614, 0.38176015, 14.0, 9.81);
        assert!(value.is_finite());
        assert!(value > 0.0);
    }
}
