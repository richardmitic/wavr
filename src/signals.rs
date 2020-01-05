extern crate hound;
extern crate sample;

#[cfg(test)]
extern crate all_asserts;
extern crate test;

use self::hound::{SampleFormat, WavReader, WavSpec};
use self::sample::types::i24::I24;
use self::sample::{conv, envelope, interpolate, ring_buffer, signal, Signal};
use std::iter;

type AudioChannel = Vec<f32>;
pub type AudioMultiChannel = Vec<AudioChannel>;
type ChannelEnvelope = Vec<f32>;
type CheckedAudioChannel = Vec<Option<f32>>;
type CheckedEnvelope = Vec<Option<f32>>;

fn deinterleave(all_samples: AudioChannel, channels: usize) -> AudioMultiChannel {
    assert!(all_samples.len() % channels == 0);
    (0..channels)
        .map(|channel| {
            all_samples
                .iter()
                .skip(channel)
                .step_by(channels)
                .map(|s| *s)
                .collect::<AudioChannel>()
        })
        .collect::<AudioMultiChannel>()
}

pub fn load_file(path: String) -> hound::Result<AudioMultiChannel> {
    let reader = WavReader::open(path)?;
    let spec = reader.spec();
    //println!("{:?}", spec);

    let all_samples: AudioChannel = match spec {
        WavSpec {
            sample_format: SampleFormat::Int,
            bits_per_sample: 16,
            ..
        } => Ok(reader
            .into_samples::<i16>()
            .map(|s| conv::i16::to_f32(s.unwrap()))
            .collect::<AudioChannel>()),
        WavSpec {
            sample_format: SampleFormat::Int,
            bits_per_sample: 24,
            ..
        } => Ok(reader
            .into_samples::<i32>()
            .map(|s| conv::i24::to_f32(I24::new_unchecked(s.unwrap())))
            .collect::<AudioChannel>()),
        WavSpec {
            sample_format: SampleFormat::Float,
            bits_per_sample: 32,
            ..
        } => Ok(reader
            .into_samples::<f32>()
            .map(|s| s.unwrap())
            .collect::<AudioChannel>()),
        _ => Err(hound::Error::Unsupported),
    }?;

    Ok(deinterleave(all_samples, spec.channels as usize))
}

pub fn extract_envelope(
    samples: &AudioChannel,
    start: f64,
    end: f64,
    envelope_length: usize,
) -> ChannelEnvelope {
    let start_frame = (samples.len() as f64 * start) as usize;
    let slice_length = (samples.len() as f64 * (end - start)) as usize;

    // An attack of 0 frames and release the length of one bin should give a decent representation
    // of the true peak value of each bin
    let attack = 0f32;
    let release = (slice_length / envelope_length) as f32;

    // Add pre-roll and and post-roll to avoid transient artifacts. Pre-roll will be removed at the
    // altered sample rate below
    let prerolled_start_frame = start_frame.checked_sub(release as usize).unwrap_or(0);
    let postrolled_end_frame = (start_frame + slice_length + release as usize).min(samples.len());
    let sig = signal::from_iter(
        samples[prerolled_start_frame..postrolled_end_frame]
            .iter()
            .map(|s| [*s]),
    );
    let detector = envelope::Detector::peak(attack, release);
    let mut envelope = sig.detect_envelope(detector);
    let interpolator = interpolate::Linear::from_source(&mut envelope);
    let converter =
        envelope.from_hz_to_hz(interpolator, slice_length as f64, envelope_length as f64);
    let preroll_frames_at_new_sample_rate = ((start_frame - prerolled_start_frame) as f32
        * (envelope_length as f32 / slice_length as f32))
        as usize;
    converter
        .until_exhausted()
        .skip(preroll_frames_at_new_sample_rate)
        .take(envelope_length)
        .map(|f| f[0])
        .collect::<AudioChannel>()
}

pub fn extract_rms(
    samples: &AudioChannel,
    start: f64,
    end: f64,
    envelope_length: usize,
) -> ChannelEnvelope {
    let start_sample = (samples.len() as f64 * start) as usize;
    let slice_length = (samples.len() as f64 * (end - start)) as usize;
    let bin_size = slice_length / envelope_length;
    let sig = signal::from_iter(
        samples[start_sample..start_sample + slice_length]
            .iter()
            .map(|s| [*s]),
    );
    let ring_buffer = ring_buffer::Fixed::from(vec![[0f32]; bin_size]);
    let mut envelope = sig.rms(ring_buffer);
    let interpolator = interpolate::Floor::from_source(&mut envelope);
    let converter =
        envelope.from_hz_to_hz(interpolator, slice_length as f64, envelope_length as f64);
    converter
        .until_exhausted()
        .take(envelope_length)
        .map(|f| f[0])
        .collect::<AudioChannel>()
}

pub fn extract_waveform(
    samples: &AudioChannel,
    start: f64,
    end: f64,
    nbins: usize,
) -> AudioChannel {
    let start_frame = (samples.len() as f64 * start) as usize;
    let slice_length = (samples.len() as f64 * (end - start)) as usize;
    let bin_length = slice_length / nbins;

    // Add pre-roll and and post-roll to avoid transient artifacts. Pre-roll will be removed at the
    // altered sample rate below
    let roll_base_value = 8;
    let prerolled_start_frame = start_frame
        .checked_sub(bin_length + roll_base_value)
        .unwrap_or(0);
    let postrolled_end_frame =
        (start_frame + slice_length + bin_length + roll_base_value).min(samples.len());
    //println!(
    //    "start_frame:{} slice_length:{} bin_length:{} preroll:{} postroll:{}",
    //    start_frame, slice_length, bin_length, prerolled_start_frame, postrolled_end_frame
    //);
    let mut sig = signal::from_iter(
        samples[prerolled_start_frame..postrolled_end_frame]
            .iter()
            .map(|s| [*s]),
    );
    let interpolator = interpolate::Linear::from_source(&mut sig);
    let converter = sig.from_hz_to_hz(interpolator, slice_length as f64, nbins as f64);
    let preroll_frames_at_new_sample_rate = ((start_frame - prerolled_start_frame) as f32
        * (nbins as f32 / slice_length as f32))
        as usize;
    //println!(
    //    "preroll_frames_at_new_sample_rate {}",
    //    preroll_frames_at_new_sample_rate
    //);
    converter
        .until_exhausted()
        .skip(preroll_frames_at_new_sample_rate)
        .take(nbins)
        .map(|f| f[0])
        .collect::<AudioChannel>()
}

fn calculate_bin_sections(start: &f64, end: &f64, nbins: &usize) -> (usize, usize, usize) {
    let before_start = start.min(0.) * -1.;
    let after_end = (end - 1.).max(0.);
    let bins_before_start = (*nbins as f64 * (before_start / (end - start))) as usize;
    let bins_after_end = (*nbins as f64 * (after_end / (end - start))) as usize;
    let centre_bins = nbins - (bins_before_start + bins_after_end);
    (bins_before_start, centre_bins, bins_after_end)
}

pub fn extract_waveform_checked(
    samples: &AudioChannel,
    start: f64,
    end: f64,
    nbins: usize,
) -> CheckedAudioChannel {
    let (bins_before, bins_centre, bins_after) = calculate_bin_sections(&start, &end, &nbins);
    let signal = extract_waveform(samples, start.max(0.), end.min(1.), bins_centre);
    iter::repeat(None)
        .take(bins_before)
        .chain(signal.into_iter().map(|x| Some(x)))
        .chain(iter::repeat(None).take(bins_after))
        .collect()
}

pub fn extract_envelope_checked(
    samples: &AudioChannel,
    start: f64,
    end: f64,
    nbins: usize,
) -> CheckedEnvelope {
    let (bins_before, bins_centre, bins_after) = calculate_bin_sections(&start, &end, &nbins);
    let signal = extract_envelope(samples, start.max(0.), end.min(1.), bins_centre);
    iter::repeat(None)
        .take(bins_before)
        .chain(signal.into_iter().map(|x| Some(x)))
        .chain(iter::repeat(None).take(bins_after))
        .collect()
}

#[cfg(test)]
mod tests {
    use self::all_asserts::{assert_gt,assert_lt};
    use self::test::Bencher;
    use super::*;
    use std::fs;

    fn write_envelope_to_file(env: &ChannelEnvelope, name: String) {
        let txt = env
            .iter()
            .map(|n| n.to_string())
            .collect::<Vec<String>>()
            .join("\n");
        fs::write(name, txt).unwrap();
    }

    #[test]
    fn extracts_envelope_of_correct_length() {
        let samples = load_file("./resources/duskwolf.wav".to_string()).unwrap();
        assert_gt!(samples.len(), 0);

        for i in (20..200).step_by(10) {
            let env = extract_envelope(&samples[0], 0., 1., i);
            assert_eq!(env.len(), i);
            //write_envelope_to_file(&env, format!("env{:03}.txt", i));
        }
    }

    #[test]
    fn extracts_rms_of_correct_length() {
        let samples = load_file("./resources/duskwolf.wav".to_string()).unwrap();
        assert_gt!(samples.len(), 0);

        for i in (20..200).step_by(10) {
            let env = extract_rms(&samples[0], 0., 1., i);
            assert_eq!(env.len(), i);
            //write_envelope_to_file(&env, format!("rms{:03}.txt", i));
        }
    }

    #[test]
    fn extracts_waveform_of_correct_length() {
        let samples = load_file("./resources/sine.c1.r8000.i16.wav".to_string()).unwrap();
        assert_gt!(samples.len(), 0);

        for i in (20..200).step_by(10) {
            let wf = extract_waveform(&samples[0], 0.1, 0.21, i);
            assert_eq!(wf.len(), i);
            write_envelope_to_file(&wf, format!("wf{:03}.txt", i));
        }
    }

    #[test]
    fn extracts_waveform_of_more_bins_than_samples() {
        let samples = load_file("./resources/sine.c1.r8000.i16.wav".to_string()).unwrap();
        assert_gt!(samples.len(), 0);

        for i in (20..200).step_by(10) {
            let wf = extract_waveform(&samples[0], 0.1, 0.11, i);
            assert_eq!(wf.len(), i);
            write_envelope_to_file(&wf, format!("wfz{:03}.txt", i));
        }
    }

    #[test]
    #[ignore]
    fn extrapolated_waveform_moves_by_fractional_sample() {
        let samples = load_file("./resources/sine.c1.r8000.i16.wav".to_string()).unwrap();
        assert_gt!(samples.len(), 0);

        for i in 0..10 {
            let start = 0.1 + i as f64 / 100.;
            let end = 0.11 + i as f64 / 100.;
            let wf = extract_waveform(&samples[0], start, end, 100);
            assert_eq!(wf.len(), 100);
            write_envelope_to_file(&wf, format!("ewmbfs{:03}.txt", i));
        }
    }

    #[test]
    #[ignore]
    fn peak_value_is_roughly_correct() {
        let samples = load_file("./resources/duskwolf.wav".to_string()).unwrap();
        let env = extract_envelope(&samples[0], 0., 1., 100);
        let max_sample_value = samples[0]
            .iter()
            .map(|s| *s)
            .map(conv::f32::to_i16)
            .max()
            .unwrap();
        let max_env_value = env.into_iter().map(conv::f32::to_i16).max().unwrap();
        assert_eq!(max_sample_value, max_env_value);
    }

    #[test]
    fn no_transient_artifacts() {
        let samples = load_file("./resources/sine.c1.r8000.f32.wav".to_string()).unwrap();
        let env = extract_envelope(&samples[0], 0.1, 0.9, 20);
        write_envelope_to_file(&env, "no_transient_artifacts.txt".to_string());
        env.into_iter().for_each(|s| {
            assert_gt!(s, 0.48);
            assert_lt!(s, 0.52);
        })
    }

    #[test]
    fn loads_files() {
        let files = vec![
            ("./resources/sine.c1.r8000.i16.wav".to_string(), 1),
            ("./resources/sine.c1.r8000.i24.wav".to_string(), 1),
            ("./resources/sine.c1.r8000.f32.wav".to_string(), 1),
            ("./resources/sine.c2.r8000.i16.wav".to_string(), 2),
            ("./resources/sine.c2.r8000.i24.wav".to_string(), 2),
            ("./resources/sine.c2.r8000.f32.wav".to_string(), 2),
        ];
        for (file, channels) in files {
            let samples = load_file(file).unwrap();
            assert_eq!(samples.len(), channels);
            assert_gt!(samples[0].len(), 0);
        }
    }

    #[test]
    fn checked_waveform_before_start() {
        let samples = load_file("./resources/sine.c1.r8000.f32.wav".to_string()).unwrap();
        let wf = extract_waveform_checked(&samples[0], -0.01, 0.01, 10);
        assert_eq!(wf.len(), 10);
        wf[0..5].iter().for_each(|x| assert!(x.is_none()));
        wf[5..].iter().for_each(|x| assert!(x.is_some()));
    }

    #[test]
    fn checked_waveform_after_end() {
        let samples = load_file("./resources/sine.c1.r8000.f32.wav".to_string()).unwrap();
        let wf = extract_waveform_checked(&samples[0], 0.99, 1.01, 10);
        assert_eq!(wf.len(), 10);
        wf[0..5].iter().for_each(|x| assert!(x.is_some()));
        wf[5..].iter().for_each(|x| assert!(x.is_none()));
    }

    #[test]
    fn checked_waveform_middle() {
        let samples = load_file("./resources/sine.c1.r8000.f32.wav".to_string()).unwrap();
        let wf = extract_waveform_checked(&samples[0], 0.1, 0.2, 10);
        assert_eq!(wf.len(), 10);
        wf.iter().for_each(|x| assert!(x.is_some()));
    }

    #[test]
    fn checked_envelope_before_start() {
        let samples = load_file("./resources/sine.c1.r8000.f32.wav".to_string()).unwrap();
        let env = extract_envelope_checked(&samples[0], -0.01, 0.01, 10);
        assert_eq!(env.len(), 10);
        env[0..5].iter().for_each(|x| assert!(x.is_none()));
        env[5..].iter().for_each(|x| assert!(x.is_some()));
    }

    #[test]
    fn checked_envelope_after_end() {
        let samples = load_file("./resources/sine.c1.r8000.f32.wav".to_string()).unwrap();
        let env = extract_envelope_checked(&samples[0], 0.99, 1.01, 10);
        assert_eq!(env.len(), 10);
        env[0..5].iter().for_each(|x| assert!(x.is_some()));
        env[5..].iter().for_each(|x| assert!(x.is_none()));
    }

    #[test]
    fn checked_envelope_middle() {
        let samples = load_file("./resources/sine.c1.r8000.f32.wav".to_string()).unwrap();
        let env = extract_waveform_checked(&samples[0], 0.1, 0.2, 10);
        assert_eq!(env.len(), 10);
        env.iter().for_each(|x| assert!(x.is_some()));
    }

    #[bench]
    fn bench_create_envelope_of_whole_file_100(b: &mut Bencher) {
        let samples = load_file("resources/duskwolf.wav".to_string()).unwrap();
        b.iter(|| extract_envelope(&samples[0], 0., 1., 100))
    }

    #[bench]
    fn bench_create_envelope_of_whole_file_1000(b: &mut Bencher) {
        let samples = load_file("resources/duskwolf.wav".to_string()).unwrap();
        b.iter(|| extract_envelope(&samples[0], 0., 1., 1000))
    }

    #[bench]
    fn bench_create_envelope_of_whole_file_10000(b: &mut Bencher) {
        let samples = load_file("resources/duskwolf.wav".to_string()).unwrap();
        b.iter(|| extract_envelope(&samples[0], 0., 1., 10000))
    }
}
