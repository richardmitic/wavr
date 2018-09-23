extern crate hound;
extern crate itertools;

use self::hound::{ WavReader };
use std::vec::Vec;

pub struct WaveForm {
    pub summary_1k: Vec<f32>,
    pub summary_8k: Vec<f32>,
    pub summary_64k: Vec<f32>,
    pub min: i32,
    pub max: i32,
    pub channels: u16
}

fn rms(signal: &[i32]) -> f32 {
    let len = signal.len() as f32;
    (signal.iter().map(|x| (x * x) as f32).sum::<f32>() / len).sqrt()
}

impl WaveForm {
    pub fn from_samples(samples: &Vec<i32>, channels: u16) -> WaveForm {
        WaveForm {
            summary_1k: samples.chunks(1024).map(|chunk| rms(chunk)).collect(),
            summary_8k: samples.chunks(8196).map(|chunk| rms(chunk)).collect(),
            summary_64k: samples.chunks(65536).map(|chunk| rms(chunk)).collect(),
            min : samples.iter().cloned().min().unwrap(),
            max : samples.iter().cloned().max().unwrap(),
            channels : channels
        }
    }

    pub fn from_file(path: &str) -> WaveForm {
        let mut reader = WavReader::open(path).unwrap();
        let samples: Vec<i32> = reader.samples::<i32>().map(|s| s.unwrap()).collect();
        let channels = reader.spec().channels;
        WaveForm::from_samples(&samples, channels)
    }
}


#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn creates_summary_from_samples() {
        let samples : Vec<i32> = vec!(0, 1, 2);
        let w = WaveForm::from_samples(&samples, 1);
        assert_eq!(w.summary_1k, [1.6666666f32]);
        assert_eq!(w.summary_8k, [1.6666666f32]);
        assert_eq!(w.summary_64k, [1.6666666f32]);
        assert_eq!(w.min, 0);
        assert_eq!(w.max, 2);
    }

    #[test]
    fn creates_summary_from_file() {
        let w = WaveForm::from_file("./resources/duskwolf.wav");
        println!("{:?}", w.summary_1k);
        println!("{:?}", w.summary_8k);
        println!("{:?}", w.summary_64k);
        assert_eq!(w.summary_1k.len(), 104);
        assert_eq!(w.summary_8k.len(), 13);
        assert_eq!(w.summary_64k.len(), 2);
        assert_eq!(w.channels, 1);
    }
}