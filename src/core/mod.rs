extern crate hound;
mod waveform;

use std;
use std::string::String;
use self::waveform::{ WaveForm, WaveSection };
use self::hound::WavReader;


fn scale(x: f64, in_min: f64, in_max: f64, out_min: f64, out_max: f64) -> f64 {
    (((out_max - out_min) * (x - in_min)) / (in_max - in_min)) + out_min
}

fn scale_vec(arr: Vec<f64>, in_min: f64, in_max: f64, out_min: f64, out_max: f64) -> Vec<f64> {
    arr.into_iter().map(|x| scale(x, in_min, in_max, out_min, out_max)).collect()
}

fn floor_n(x: usize, n: usize) -> usize {
    (x / n) * n
}

fn split_into_channels(samples: Vec<f32>, n: usize) -> Vec<Vec<f32>> {
    (0..n).map(|chan| {
        samples.iter()
            .skip(chan)
            .step_by(n)
            .map(|s_ptr| *s_ptr)
            .collect::<Vec<f32>>()
    }).collect::<Vec<Vec<f32>>>()
}

type WavePeaksChannel = Vec<Option<WaveSection>>;

pub struct DisplayChars {
    rms: char,
    peak: char,
    zero: char,
    none: char,
    sample: char
}

pub struct Core {
    file: String,
    summary: Option<WaveForm>,
    chars: DisplayChars
}

impl Core {
    pub fn new() -> Core {
        Core {
            file: String::new(),
            summary: Option::None,
            chars: DisplayChars {
                rms: 'o',
                peak: '·',
                zero: '=',
                none: '-',
                sample: '•'
            }
        }
    }

    pub fn load(&mut self, f: String) {
        self.file = f;
        self.summary = Some(WaveForm::from_file(&self.file));
    }

    pub fn get_peaks(&mut self, start: &f64, end: &f64, num_peaks: u32) -> WavePeaksChannel {
        let max_points = self.summary.as_ref().unwrap().summary_64.len() - 1;
        let skip = (*end - *start) / (num_peaks as f64);

        (0..num_peaks).map(|x| {
            let phase = *start + (x as f64 * skip);
            match phase {
                p if p < 0f64 => None,
                p if p >= 1f64 => None,
                _ => {
                    let interp_index = phase * max_points as f64;
                    let int_index = interp_index as usize;
                    let coeff = interp_index - interp_index.floor();
                    let x = self.summary.as_ref().unwrap().summary_64[int_index].clone();
                    let y = self.summary.as_ref().unwrap().summary_64[int_index + 1].clone();
                    let diff = y - x;
                    Some(x + (diff * coeff as f32))
                }
            }
        }).collect::<WavePeaksChannel>()
    }

    pub fn draw_wave(&mut self, peaks: Vec<Option<WaveSection>>, width: &usize, height: &usize) -> Vec<Vec<char>> {
        let mut arr = vec![vec![' '; *width]; *height];
        peaks.iter().enumerate().for_each(|(i, sect)| {
            let full_scale_max = (std::i16::MAX) as f64;
            let full_scale_min = (std::i16::MIN) as f64;
            let this_scale_max = (*height - 1) as f64;
            let centre_row = scale(0., full_scale_min, full_scale_max, this_scale_max, 0.) as usize;
            match *sect {
                Some(ws) => {
                    let max_scaled = scale(ws.max as f64, full_scale_min, full_scale_max, this_scale_max, 0.).max(0f64) + 0.5;
                    let min_scaled = scale(ws.min as f64, full_scale_min, full_scale_max, this_scale_max, 0.).min(this_scale_max) + 0.5;
                    let max_rms_scaled = scale(ws.rms as f64, full_scale_min, full_scale_max, this_scale_max, 0.).max(0f64) + 0.5;
                    let min_rms_scaled = scale(-(ws.rms as f64), full_scale_min, full_scale_max, this_scale_max, 0.).min(this_scale_max) + 0.5;
                    //println!("{} {} {}", max_scaled, min_scaled, mid_point);
                    let max_idx = max_scaled as usize;
                    let min_idx = min_scaled as usize;
                    let max_rms_idx = max_rms_scaled as usize;
                    let min_rms_idx = min_rms_scaled as usize;
                    match min_idx == max_idx {
                        true => arr[max_idx][i] = self.chars.zero,
                        false => {
                            for j in max_idx..min_idx {
                                arr[j][i] = self.chars.peak;
                            }

                            for j in max_rms_idx..min_rms_idx {
                                arr[j][i] = self.chars.rms;
                            }
                        }
                    }
                },
                None => arr[centre_row][i] = self.chars.none
            }
        });

        arr
    }


//    pub fn get_samples_multichannel(&mut self, start: &f64, end: &f64, num_bins: usize) -> Vec<Vec<f32>> {
//        let mut reader = WavReader::open(self.file.as_str()).unwrap();
//        let spec = reader.spec();
//
//        let clipped_start = (*start).min(1.).max(0.);
//        let clipped_end = (*end).min(1.).max(0.);
//        let full_len = reader.len() as f64;
//        let start_frame = floor_n(((full_len - 1.) * clipped_start) as usize, spec.channels as usize);
//        let end_frame = floor_n(((full_len - 1.) * clipped_end) as usize, spec.channels as usize);
//        let num_frames = end_frame - start_frame;
//
//        if num_frames == 0 {
//            return vec![vec![0f32; num_bins]; spec.channels as usize]
//        }
//
//        let _pos = reader.seek(start_frame as u32);
//        let section: Vec<i32> = reader.samples::<i32>()
//            .take(num_frames)
//            .map(|s| s.unwrap())
//            .collect();
//
//
//        (0..num_bins).map(|n| {
//            let interp_index = scale(n as f64, 0., num_bins as f64, 0., num_frames as f64);
//            //println!("{} {} {}", n, section.len(), interp_index);
//            match interp_index {
//                ii if ii < 0f64 => 0f32,
//                ii if ii >= num_frames as f64 => 0f32,
//                _ => {
//                    let int_index = interp_index as usize;
//                    section[int_index] as f32
//                }
//            }
//
//        }).collect::<Vec<f32>>()
//    }

    pub fn get_samples(&mut self, start: &f64, end: &f64, num_bins: usize) -> Vec<Option<f32>> {
        let mut reader = WavReader::open(self.file.as_str()).unwrap();
        let _spec = reader.spec();

        let full_len = reader.duration() as f64;
        let bin_indices = (0..num_bins).map(|x| x as f64).collect();
        let interp_indices = scale_vec(bin_indices, 0., num_bins as f64 - 1., *start * (full_len - 0.), *end * (full_len - 1.));
        let start_frame = interp_indices[0].min(full_len - 1.).max(0.) as usize;
        let end_frame = interp_indices[num_bins - 1].ceil().min(full_len - 1.).max(0.) as usize;
 
        let num_frames = end_frame - start_frame;
        if num_frames == 0 {
            return vec![None; num_bins]
        }

        let _pos = reader.seek(start_frame as u32);
        let section = reader.samples::<i32>()
            .take(num_frames)
            .map(|s| s.unwrap())
            .collect::<Vec<i32>>();
        assert_eq!(num_frames, section.len());

        interp_indices.into_iter().map(|x| {
            match x {
                ii if ii < 0. => None,
                ii if ii >= full_len - 1. => None,
                _ => {
                    let idx = x as usize - start_frame;
                    Some(section[idx] as f32)
                }
            }
        }).collect::<Vec<Option<f32>>>()
    }

    pub fn draw_samples(&mut self, samples: Vec<Option<f32>>, width: &usize, height: &usize) -> Vec<Vec<char>> {
        let mut arr = vec![vec![' '; *width]; *height];
        let full_scale_max = (std::i16::MAX) as f64;
        let full_scale_min = (std::i16::MIN) as f64;
        let this_scale_max = (*height - 1) as f64;
        samples.into_iter().enumerate().for_each(|(i, sample)| {
            match sample {
                None => {
                    let col = scale(0., full_scale_min, full_scale_max, this_scale_max, 0.) as usize;
                    arr[col][i] = self.chars.none;
                },
                Some(x) => {
                    let col = scale(x as f64, full_scale_min, full_scale_max, this_scale_max, 0.) as usize;
                    arr[col][i] = self.chars.sample;
                }
            }
        });

        arr
    }

    pub fn should_draw_samples(&mut self, start: &f64, end: &f64, width: &usize) -> bool {
        let range = *end - *start;
        let num_peaks = self.summary.as_ref().unwrap().summary_64.len() as f64 * range;
        num_peaks < (width / 2) as f64
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn loads_file() {
        let mut c = Core::new();
        c.load("./resources/duskwolf.wav".to_string());
        let p = c.get_peaks(&0., &1., 20);
        assert_eq!(p.len(), 20);
    }

    #[test]
    fn draws_wave() {
        let mut c = Core::new();
        c.load("./resources/duskwolf.wav".to_string());
        let p = c.get_peaks(&0., &1., 120);
        let w = c.draw_wave(p, &120, &30);
        w.iter().for_each(|row: &Vec<char>| println!("{:?}", row.iter().collect::<String>()));
    }

    #[test]
    fn gets_samples() {
        let mut c = Core::new();
        c.load("./resources/duskwolf.wav".to_string());
        let s = c.get_samples(&0.3, &0.3015, 30);
        //println!("{:?}", s);
        assert_eq!(s.len(), 30);
    }

    #[test]
    fn gets_samples_2() {
        let mut c = Core::new();
        c.load("./resources/duskwolf.wav".to_string());
        let s = c.get_samples(&0.3, &0.3015, 164);
        //println!("{:?}", s);
        assert_eq!(s.len(), 164);
    }

    #[test]
    fn draws_samples_big() {
        let mut c = Core::new();
        c.load("./resources/duskwolf.wav".to_string());
        let s = c.get_samples(&0.1, &0.9, 120);
        let w = c.draw_samples(s, &120, &60);
    }

    #[test]
    fn interpolate_samples_when_getting_short_block() {
        let mut c = Core::new();
        c.load("./resources/sine.wav".to_string());
        let s = c.get_samples(&0., &0.01, 20);
        let expected = [
            Some(0.0), Some(0.0), Some(0.0), 
            Some(0.0), Some(214.0), Some(214.0), Some(214.0), Some(214.0), 
            Some(430.0), Some(430.0), Some(430.0), Some(430.0),
            Some(642.0), Some(642.0), Some(642.0), Some(642.0),
            Some(858.0), Some(858.0), Some(858.0), Some(858.0)
        ];
        assert_eq!(s, expected);
    }

    #[test]
    fn draws_samples() {
        let mut c = Core::new();
        c.load("./resources/duskwolf.wav".to_string());
        let s = c.get_samples(&0.3, &0.3015, 30);
        let w = c.draw_samples(s, &120, &30);
        assert_eq!(w.len(), 30);
        assert_eq!(w[0].len(), 120);
    }

    #[test]
    fn out_of_bounds_peaks_are_zero() {
        let mut c = Core::new();
        c.load("./resources/duskwolf.wav".to_string());
        let p = c.get_peaks(&-1., &0., 5);
        assert_eq!(p.iter().fold(true, |a, b| a && b.is_none()), true);
        let p = c.get_peaks(&1., &2., 5);
        assert_eq!(p.iter().fold(true, |a, b| a && b.is_none()), true);
    }

    #[test]
    fn panic1() {
        let mut c = Core::new();
        c.load("./resources/duskwolf.wav".to_string());
        let p = c.get_peaks(&0.074016, &0.401696, 181);
        let w = c.draw_wave(p, &181, &30);
        assert_eq!(w.len(), 30);
    }

    #[test]
    fn panic2() {
        let mut c = Core::new();
        c.load("/Users/richardmitic/Music/mmm.wav".to_string());
        let p = c.get_samples(&0.48874100093157374, &0.5168884986026394, 204);
        let w = c.draw_samples(p, &204, &30);
        assert_eq!(w.len(), 30);
    }


    #[test]
    fn gets_correct_samples() {
        let mut c = Core::new();
        c.load("./resources/ramp.wav".to_string());
        let p = c.get_samples(&0., &(100. / 1024.), 100);
        println!("{:?}", p);
        for i in 0..100 {
            assert_close_enough(p[i].unwrap() as f64, i as f64, 0.0000001);
        }
    }

    #[test]
    fn gets_correct_samples_middle() {
        let mut c = Core::new();
        c.load("./resources/ramp.wav".to_string());
        for j in 1..10 {
            let start = (j as f64) * 100.;
            let end = start + 100.;
            let p = c.get_samples(&(start / 1024.), &(end / 1024.), 100);
            assert_eq!(p.len(), 100);
            println!("{:?}", p);
            for i in 0..100 {
                assert_close_enough(p[i].unwrap() as f64, i as f64 + start, 0.0000001);
            }
        }
    }

    #[test]
    fn out_of_bounds_samples_are_zero() {
        let mut c = Core::new();
        c.load("./resources/sine.wav".to_string());
        let p = c.get_samples(&-1., &0., 5);
        assert_eq!(p, [None, None, None, None, None]);
        let p = c.get_samples(&1., &2., 5);
        assert_eq!(p, [None, None, None, None, None]);
        let p = c.get_samples(&0.5, &1.5, 5);
        assert_ne!(p[0], None);
        assert_ne!(p[1], None);
        assert_eq!(p[2], None);
        assert_eq!(p[3], None);
        assert_eq!(p[4], None);
    }

    fn assert_close_enough(x: f64, y: f64, epsilon: f64) {
        assert_eq!((x - y).abs() < epsilon, true);
    }

    #[test]
    fn scale_up() {
        assert_close_enough(scale(0.1, -1., 1., -2., 2.), 0.2, 0.00000001);
    }

    #[test]
    fn scale_invert() {
        assert_close_enough(scale(0.1, -1., 1., 2., -2.), -0.2, 0.00000001);
    }

    #[test]
    fn scale_shift() {
        assert_close_enough(scale(0.1, -1., 1., 0., 2.), 1.1, 0.00000001);
    }

    #[test]
    fn scale_shift_invert() {
        assert_close_enough(scale(0.1, -1., 1., 2., 0.), 0.9, 0.00000001);
    }

    #[test]
    fn split_stero() {
        let samples: Vec<f32> = (0..8).into_iter().map(|i| i as f32).collect();
        let channels = split_into_channels(samples, 2);
        assert_eq!(channels[0], vec![0f32,2f32,4f32,6f32]);
        assert_eq!(channels[1], vec![1f32,3f32,5f32,7f32]);
    }

    #[test]
    fn mono_samples_reshaped() {
        let samples: Vec<f32> = (0..8).into_iter().map(|i| i as f32).collect();
        let channels = split_into_channels(samples.clone(), 1);
        assert_eq!(channels[0], samples);
    }
}
