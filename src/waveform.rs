extern crate hound;
extern crate itertools;
extern crate rustfft;

use self::hound::WavReader;
use self::rustfft::num_complex::Complex;
use self::rustfft::num_traits::Zero;
use self::rustfft::FFTplanner;
use pcm::read_i16_section;
use std::ops::{Add, Mul, Sub};
use std::vec::Vec;
use util::FileType;

pub type SpectralFrame = Vec<f32>;

#[derive(Debug, Clone, Copy)]
pub struct WaveSection {
    pub min: f32,
    pub max: f32,
    pub rms: f32,
}

impl WaveSection {
    pub fn from_signal_multichannel(signal: &[i16], channels: u16) -> Vec<WaveSection> {
        (0..channels)
            .map(|n| WaveSection {
                min: *signal
                    .iter()
                    .skip(n as usize)
                    .step_by(channels as usize)
                    .min()
                    .unwrap() as f32,
                max: *signal
                    .iter()
                    .skip(n as usize)
                    .step_by(channels as usize)
                    .max()
                    .unwrap() as f32,
                rms: rms(signal, n as usize, channels as usize),
            })
            .collect()
    }
}

impl Sub for WaveSection {
    type Output = WaveSection;

    fn sub(self, other: WaveSection) -> WaveSection {
        WaveSection {
            min: self.min - other.min,
            max: self.max - other.max,
            rms: self.rms - other.rms,
        }
    }
}

impl Add for WaveSection {
    type Output = WaveSection;

    fn add(self, other: WaveSection) -> WaveSection {
        WaveSection {
            min: self.min + other.min,
            max: self.max + other.max,
            rms: self.rms + other.rms,
        }
    }
}

impl Mul<f32> for WaveSection {
    type Output = WaveSection;

    fn mul(self, rhs: f32) -> WaveSection {
        WaveSection {
            min: self.min * rhs,
            max: self.max * rhs,
            rms: self.rms * rhs,
        }
    }
}

fn complex_from_signal_multichannel(signal: &[i16], channels: u16) -> Vec<Vec<Complex<f32>>> {
    (0..channels as usize)
        .map(|n| {
            signal
                .iter()
                .cloned()
                .skip(n)
                .step_by(channels as usize)
                .map(|sample| Complex::<f32> {
                    re: sample as f32,
                    im: 0f32,
                })
                .collect::<Vec<Complex<f32>>>()
        })
        .collect::<Vec<Vec<Complex<f32>>>>()
}

pub struct WaveForm {
    pub summary_256: Vec<Vec<WaveSection>>,
    pub summary_1k: Vec<Vec<WaveSection>>,
    pub summary_8k: Vec<Vec<WaveSection>>,
    pub summary_256k: Vec<Vec<WaveSection>>,
    pub spectrum: Vec<Vec<SpectralFrame>>,
    pub min: i16,
    pub max: i16,
    pub channels: u16,
    pub length: usize,
}

fn rms(signal: &[i16], offset: usize, step: usize) -> f32 {
    let len = signal.len() as f32;
    (signal
        .iter()
        .skip(offset)
        .step_by(step)
        .map(|x| {
            let f = *x as f32;
            f * f
        })
        .sum::<f32>()
        / len)
        .sqrt()
}

impl WaveForm {
    pub fn make_summary(
        samples: &Vec<i16>,
        chunk_size: usize,
        channels: u16,
    ) -> Vec<Vec<WaveSection>> {
        let mut summary = vec![vec![]; channels as usize];
        for chunk in samples.chunks(chunk_size) {
            let ws = WaveSection::from_signal_multichannel(chunk, channels);
            for i in 0..channels as usize {
                summary[i].push(ws[i]);
            }
        }
        summary
    }

    pub fn make_spectrum(
        samples: &Vec<i16>,
        fft_size: usize,
        channels: u16,
    ) -> Vec<Vec<SpectralFrame>> {
        let mut spectrum = vec![vec![]; channels as usize];
        let mut planner = FFTplanner::new(false);
        let fft = planner.plan_fft(fft_size);
        let mut output: Vec<Complex<f32>> = vec![Complex::zero(); fft_size];
        let window = apodize::hanning_iter(fft_size).collect::<Vec<f64>>();
        for chunk in samples.chunks(fft_size * channels as usize) {
            let input = complex_from_signal_multichannel(chunk, channels);
            if input[0].len() != fft_size {
                // Ignore incomplete frames
                continue;
            }
            for i in 0..channels as usize {
                let mut windowed_input = input[i]
                    .iter()
                    .zip(window.iter())
                    .map(|(x, w)| x.scale(*w as f32))
                    .collect::<Vec<Complex<f32>>>();
                fft.process(&mut windowed_input, &mut output);
                let frame_abs = output.iter().map(Complex::norm).collect::<SpectralFrame>();
                spectrum[i].push(frame_abs);
            }
        }
        spectrum
    }

    pub fn from_samples(samples: &Vec<i16>, channels: u16) -> WaveForm {
        WaveForm {
            summary_256: WaveForm::make_summary(samples, 64, channels),
            summary_1k: WaveForm::make_summary(samples, 1024, channels),
            summary_8k: WaveForm::make_summary(samples, 8196, channels),
            summary_256k: WaveForm::make_summary(samples, 65536, channels),
            spectrum: WaveForm::make_spectrum(samples, 256, channels),
            min: samples.iter().cloned().min().unwrap(),
            max: samples.iter().cloned().max().unwrap(),
            channels: channels,
            length: samples.len(),
        }
    }

    pub fn from_file(path: &str, ft: &FileType, channels: Option<u16>) -> WaveForm {
        match ft {
            &FileType::WAV => WaveForm::from_wav_file(path),
            _ => WaveForm::from_pcm_file(path, channels.unwrap_or(1)),
        }
    }

    fn from_wav_file(path: &str) -> WaveForm {
        let mut reader = WavReader::open(path).unwrap();
        let samples: Vec<i16> = reader.samples::<i16>().map(|s| s.unwrap()).collect();
        let channels = reader.spec().channels;
        WaveForm::from_samples(&samples, channels)
    }

    fn from_pcm_file(path: &str, channels: u16) -> WaveForm {
        let samples = read_i16_section(path, 0, None);
        WaveForm::from_samples(&samples, channels)
    }

    pub fn get_best_summary(
        &self,
        start: &f64,
        end: &f64,
        num_peaks: &u32,
    ) -> &Vec<Vec<WaveSection>> {
        let samples_per_bin = ((*end - *start) * self.length as f64) / (*num_peaks as f64);
        let length_differences = vec![
            (256f64 - samples_per_bin).abs(),
            (1024f64 - samples_per_bin).abs(),
            (8196f64 - samples_per_bin).abs(),
            (65536f64 - samples_per_bin).abs(),
        ];

        let mut min_index = 0;
        let mut min = length_differences[min_index];

        for (i, v) in length_differences.iter().enumerate() {
            if min > *v {
                min_index = i;
                min = *v;
            }
        }

        match min_index {
            3 => &self.summary_256k,
            2 => &self.summary_8k,
            1 => &self.summary_1k,
            _ => &self.summary_256,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn creates_summary_from_samples() {
        let samples: Vec<i16> = vec![0, 1, 2];
        let w = WaveForm::from_samples(&samples, 1);
        assert_eq!(w.min, 0);
        assert_eq!(w.max, 2);
    }

    #[test]
    fn creates_summary_from_file() {
        let w = WaveForm::from_file("./resources/duskwolf.wav", &FileType::WAV, None);
        assert_eq!(w.summary_1k.len(), 1);
        assert_eq!(w.summary_1k[0].len(), 104);
        assert_eq!(w.summary_8k.len(), 1);
        assert_eq!(w.summary_8k[0].len(), 13);
        assert_eq!(w.summary_256k.len(), 1);
        assert_eq!(w.summary_256k[0].len(), 2);
        assert_eq!(w.spectrum.len(), 1);
        assert_eq!(w.spectrum[0].len(), 412);
        assert_eq!(w.spectrum[0][0].len(), 256);
        assert_eq!(w.channels, 1);
    }

    #[test]
    fn creates_summary_from_pcm_file() {
        let w = WaveForm::from_file("./resources/stereo.pcm", &FileType::PCM, Some(1));
        assert_eq!(w.summary_1k.len(), 1);
        assert_eq!(w.summary_1k[0].len(), 1);
    }

    #[test]
    fn chooses_correct_summary() {
        let w = WaveForm::from_file("./resources/duskwolf.wav", &FileType::WAV, None);
        let x1 = w.get_best_summary(&0., &1., &100);
        let x2 = w.get_best_summary(&0.5, &0.6, &100);
        assert_eq!(x1[0].len(), w.summary_1k[0].len());
        assert_eq!(x2[0].len(), w.summary_256[0].len());
    }

    #[test]
    fn wave_section_stereo() {
        let samples: Vec<i16> = vec![0, 1, 2, 3, 4, 5, 6, 7];
        let ws = WaveSection::from_signal_multichannel(&samples, 2);
        assert_eq!(ws.len(), 2);
        assert_eq!(ws[0].min, 0.);
        assert_eq!(ws[1].min, 1.);
        assert_eq!(ws[0].max, 6.);
        assert_eq!(ws[1].max, 7.);
        assert_eq!(ws[0].rms, 2.6457512);
        assert_eq!(ws[1].rms, 3.2403703);
    }
}
