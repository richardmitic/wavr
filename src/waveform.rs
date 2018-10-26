extern crate hound;
extern crate itertools;

use std::ops::{ Add, Mul, Sub };
use self::hound::{ WavReader };
use std::vec::Vec;
use pcm::read_i16_section;
use util::FileType;


#[derive(Debug, Clone, Copy)]
pub struct WaveSection {
    pub min: f32,
    pub max: f32,
    pub rms: f32
}

impl WaveSection {
    pub fn from_signal(signal: &[i16]) -> WaveSection {
        WaveSection {
            min: *signal.iter().min().unwrap() as f32,
            max: *signal.iter().max().unwrap() as f32,
            rms: rms(signal)
        }
    }
}

impl Sub for WaveSection {
    type Output = WaveSection;

    fn sub(self, other: WaveSection) -> WaveSection {
        WaveSection {
            min: self.min - other.min,
            max: self.max - other.max,
            rms: self.rms - other.rms
        }
    }
}

impl Add for WaveSection {
    type Output = WaveSection;

    fn add(self, other: WaveSection) -> WaveSection {
        WaveSection {
            min: self.min + other.min,
            max: self.max + other.max,
            rms: self.rms + other.rms
        }
    }
}

impl Mul<f32> for WaveSection {
    type Output = WaveSection;

    fn mul(self, rhs: f32) -> WaveSection {
        WaveSection {
            min: self.min * rhs,
            max: self.max * rhs,
            rms: self.rms * rhs
        }
    }
}


pub struct WaveForm {
    pub summary_64: Vec<WaveSection>,
    pub summary_1k: Vec<WaveSection>,
    pub summary_8k: Vec<WaveSection>,
    pub summary_64k: Vec<WaveSection>,
    pub min: i16,
    pub max: i16,
    pub channels: u16,
}

fn rms(signal: &[i16]) -> f32 {
    let len = signal.len() as f32;
    (signal.iter().map(|x| {
        let f = *x as f32;
        f * f
    }).sum::<f32>() / len).sqrt()
}

impl WaveForm {
    pub fn from_samples(samples: &Vec<i16>, channels: u16) -> WaveForm {
        WaveForm {
            summary_64: samples.chunks(64).map(|chunk| WaveSection::from_signal(chunk)).collect(),
            summary_1k: samples.chunks(1024).map(|chunk| WaveSection::from_signal(chunk)).collect(),
            summary_8k: samples.chunks(8196).map(|chunk| WaveSection::from_signal(chunk)).collect(),
            summary_64k: samples.chunks(65536).map(|chunk| WaveSection::from_signal(chunk)).collect(),
            min : samples.iter().cloned().min().unwrap(),
            max : samples.iter().cloned().max().unwrap(),
            channels : channels
        }
    }

    pub fn from_file(path: &str, ft: &FileType, channels: Option<u16>) -> WaveForm {
         match ft {
            &FileType::WAV => WaveForm::from_wav_file(path),
            &FileType::PCM => WaveForm::from_pcm_file(path, channels.unwrap_or(1))
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
}


#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn creates_summary_from_samples() {
        let samples : Vec<i16> = vec!(0, 1, 2);
        let w = WaveForm::from_samples(&samples, 1);
        assert_eq!(w.min, 0);
        assert_eq!(w.max, 2);
    }

    #[test]
    fn creates_summary_from_file() {
        let w = WaveForm::from_file("./resources/duskwolf.wav", &FileType::WAV, None);
        assert_eq!(w.summary_1k.len(), 104);
        assert_eq!(w.summary_8k.len(), 13);
        assert_eq!(w.summary_64k.len(), 2);
        assert_eq!(w.channels, 1);
    }

    #[test]
    fn creates_summary_from_pcm_file() {
        let w = WaveForm::from_file("./resources/stereo.pcm", &FileType::PCM, Some(1));
        assert_eq!(w.summary_1k.len(), 1);
    }
}