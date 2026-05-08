use std::f32::consts::TAU;

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

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BuildComponentsError {
    InvalidComponentCount,
    BufferTooSmall,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ComputeWaveError {
    InvalidFrameCount,
    InvalidComponentCount,
    BufferTooSmall,
}

struct AxisRng {
    state: u64,
}

impl AxisRng {
    fn new(seed: u64) -> Self {
        Self { state: seed.max(1) }
    }

    fn next_u32(&mut self) -> u32 {
        let mut x = self.state;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.state = x;
        ((x.wrapping_mul(0x2545_f491_4f6c_dd1d)) >> 32) as u32
    }

    fn next_f32(&mut self) -> f32 {
        const SCALE: f32 = 1.0 / ((1u32 << 24) as f32);
        ((self.next_u32() >> 8) as f32) * SCALE
    }

    fn standard_normal_f32(&mut self) -> f32 {
        let u1 = self.next_f32().max(f32::MIN_POSITIVE);
        let u2 = self.next_f32();
        (-2.0 * u1.ln()).sqrt() * (TAU * u2).cos()
    }
}

fn normalize2(x: f32, y: f32) -> (f32, f32) {
    let len = (x * x + y * y).sqrt();
    if len < 1e-6 {
        (0.0, 0.0)
    } else {
        (x / len, y / len)
    }
}

#[allow(clippy::too_many_arguments)]
pub fn build_components(
    kx: &mut [f32],
    ky: &mut [f32],
    omega: &mut [f32],
    amp: &mut [f32],
    phase0: &mut [f32],
    component_count: usize,
    wind_x: f32,
    wind_y: f32,
    wind_speed: f32,
    gravity: f32,
    amplitude_scale: f32,
    seed: u64,
) -> Result<(), BuildComponentsError> {
    if component_count == 0 || component_count % 2 != 0 {
        return Err(BuildComponentsError::InvalidComponentCount);
    }

    if kx.len() < component_count
        || ky.len() < component_count
        || omega.len() < component_count
        || amp.len() < component_count
        || phase0.len() < component_count
    {
        return Err(BuildComponentsError::BufferTooSmall);
    }

    let mut rng = AxisRng::new(seed);
    let (wind_x, wind_y) = normalize2(wind_x, wind_y);
    let pair_count = component_count / 2;
    let base_angle = wind_y.atan2(wind_x);
    let mut idx = 0;

    for i in 0..pair_count {
        let band = if pair_count <= 1 {
            0.0
        } else {
            i as f32 / (pair_count - 1) as f32
        };
        let wavelength = 1.2 + 9.0 * band.powi(2);
        let k = TAU / wavelength;
        let angle = base_angle + rng.standard_normal_f32() * 1.05 * (0.2 + 0.8 * band);

        for (dir, scale) in [(1.0, 1.0), (-1.0, 0.45)] {
            let wave_kx = dir * angle.cos() * k;
            let wave_ky = dir * angle.sin() * k;
            let spec = phillips_spectrum(wave_kx, wave_ky, wind_x, wind_y, wind_speed, gravity);

            amp[idx] =
                amplitude_scale * scale * spec.max(0.0).sqrt() * (0.35 + 0.65 * (1.0 - band));
            phase0[idx] = rng.next_f32() * TAU;
            omega[idx] = (gravity * k).sqrt();
            kx[idx] = wave_kx;
            ky[idx] = wave_ky;
            idx += 1;
        }
    }

    Ok(())
}

pub fn compute_wave(
    frame: &mut [f32],
    phase_base: &[f32],
    omega: &[f32],
    amp: &[f32],
    phase0: &[f32],
    frame_count: usize,
    component_count: usize,
    time: f32,
) -> Result<(), ComputeWaveError> {
    if frame_count == 0 {
        return Err(ComputeWaveError::InvalidFrameCount);
    }
    if component_count == 0 {
        return Err(ComputeWaveError::InvalidComponentCount);
    }

    let phase_base_len = frame_count
        .checked_mul(component_count)
        .ok_or(ComputeWaveError::BufferTooSmall)?;

    if frame.len() < frame_count
        || phase_base.len() < phase_base_len
        || omega.len() < component_count
        || amp.len() < component_count
        || phase0.len() < component_count
    {
        return Err(ComputeWaveError::BufferTooSmall);
    }

    for idx in 0..frame_count {
        let mut height = 0.0;
        for component in 0..component_count {
            let phase_index = idx + component * frame_count;
            height += amp[component]
                * (phase_base[phase_index] - omega[component] * time + phase0[component]).cos();
        }
        frame[idx] = height;
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::{build_components, compute_wave, phillips_spectrum};
    use std::f32::consts::TAU;

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

    #[test]
    fn builds_components_into_soa_buffers() {
        const COUNT: usize = 128;
        let mut kx = vec![0.0; COUNT];
        let mut ky = vec![0.0; COUNT];
        let mut omega = vec![0.0; COUNT];
        let mut amp = vec![0.0; COUNT];
        let mut phase0 = vec![0.0; COUNT];

        build_components(
            &mut kx,
            &mut ky,
            &mut omega,
            &mut amp,
            &mut phase0,
            COUNT,
            0.92,
            0.38,
            14.0,
            9.81,
            0.08,
            42,
        )
        .unwrap();

        assert!(kx.iter().any(|value| *value != 0.0));
        assert!(ky.iter().any(|value| *value != 0.0));
        assert!(omega.iter().all(|value| value.is_finite() && *value > 0.0));
        assert!(amp.iter().all(|value| value.is_finite() && *value >= 0.0));
        assert!(
            phase0
                .iter()
                .all(|value| value.is_finite() && *value >= 0.0 && *value < TAU)
        );
    }

    #[test]
    fn computes_wave_frame_from_phase_components() {
        let phase_base = vec![0.0, 1.0, 2.0, 3.0];
        let omega = vec![1.0, 2.0];
        let amp = vec![0.5, 0.25];
        let phase0 = vec![0.0, 0.5];
        let mut frame = vec![0.0; 2];

        compute_wave(&mut frame, &phase_base, &omega, &amp, &phase0, 2, 2, 0.125).unwrap();

        assert!(frame.iter().all(|value| value.is_finite()));
        assert_ne!(frame[0], 0.0);
        assert_ne!(frame[1], 0.0);
    }
}
