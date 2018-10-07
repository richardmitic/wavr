extern crate hound;
mod waveform;

use std::string::String;
use self::waveform::{ WaveForm, WaveSection };
use std;
use self::hound::WavReader;

fn scale(x: f64, in_min: f64, in_max: f64, out_min: f64, out_max: f64) -> f64 {
    (((out_max - out_min) * (x - in_min)) / (in_max - in_min)) + out_min
}

pub struct Core {
    file: String,
    summary: Option<WaveForm>,
}

impl Core {
    pub fn new() -> Core {
        Core {
            file: String::new(),
            summary: Option::None,
        }
    }

    pub fn load(&mut self, f: String) {
        self.file = f;
        self.summary = Some(WaveForm::from_file(&self.file)); 
    }


    pub fn get_peaks_extra(&mut self, start: &f64, end: &f64, num_peaks: u32) -> Vec<Option<WaveSection>> {
        let max_points = self.summary.as_ref().unwrap().summary_64_extra.len() - 1;
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
                    let x = self.summary.as_ref().unwrap().summary_64_extra[int_index].clone();
                    let y = self.summary.as_ref().unwrap().summary_64_extra[int_index + 1].clone();
                    let diff = y - x;
                    Some(x + (diff * coeff as f32))
                }
            }
        }).collect::<Vec<Option<WaveSection>>>()
    }

    pub fn draw_wave_extra(&mut self, peaks: Vec<Option<WaveSection>>, width: &usize, height: &usize) -> Vec<Vec<char>> {
        let mut arr = vec![vec![' '; *width]; *height];
        peaks.iter().enumerate().for_each(|(i, sect)| {
            let centre_row = (*height / 2) - 1;
            let mid_point = (*height as f64) / 2.;
            match *sect {
                Some(ws) => {
                    let full_scale_max = (std::i16::MAX / 2) as f64;
                    let full_scale_min = (std::i16::MIN / 2) as f64;
                    let this_scale_max = *height as f64 - 0.01;
                    let max_scaled = scale(ws.max as f64, full_scale_min, full_scale_max, this_scale_max, 0.).max(0f64);
                    let min_scaled = scale(ws.min as f64, full_scale_min, full_scale_max, this_scale_max, 0.).min(this_scale_max);
                    let max_rms_scaled = scale(ws.rms as f64, full_scale_min, full_scale_max, this_scale_max, 0.).max(0f64);
                    let min_rms_scaled = scale(-(ws.rms as f64), full_scale_min, full_scale_max, this_scale_max, 0.).min(this_scale_max);
                    //println!("{} {} {}", max_scaled, min_scaled, mid_point);
                    let max_idx = max_scaled as usize;
                    let min_idx = min_scaled as usize;
                    let max_rms_idx = max_rms_scaled as usize;
                    let min_rms_idx = min_rms_scaled as usize;
                    match min_idx == max_idx {
                        true => arr[max_idx][i] = '=',
                        false => {
                            for j in max_idx..min_idx { 
                                arr[j][i] = '·'; 
                            }
                            
                            for j in max_rms_idx..min_rms_idx { 
                                arr[j][i] = 'o'; 
                            }
                        }
                    }
                },
                None => arr[centre_row][i] = '-'
            }
        });

        arr
    }

    /// Returns a set of wave peaks resampled to the given length
    /// start and end are in the range 0. to 1.
    pub fn get_peaks(&mut self, start: f32, end: f32, num_peaks: u32) -> Vec<f32> {
        let max_points = self.summary.as_ref().unwrap().summary_64.len() - 1;
        let skip = (end - start) / (num_peaks as f32);
        (0..num_peaks).map(|x| {
            let phase = start + (x as f32 * skip);
            match phase {
                p if p < 0f32 => 0f32,
                p if p >= 1f32 => 0f32,
                _ => {
                    let interp_index = phase * max_points as f32;
                    let int_index = interp_index as usize;
                    let coeff = interp_index - interp_index.floor();
                    let x = self.summary.as_ref().unwrap().summary_64[int_index];
                    let y = self.summary.as_ref().unwrap().summary_64[int_index + 1];
                    let diff = y - x;
                    x + (diff * coeff)

                }
            }
        }).collect::<Vec<f32>>()
    }

    pub fn draw_wave(&mut self, peaks: Vec<f32>, width: &usize, height: &usize) -> Vec<Vec<char>> {
        let mut arr = vec![vec![' '; *width]; *height];
        peaks.iter().enumerate().for_each(|(i, peak)| {
            let centre_row = *height / 2;
            let spread = (*height * *peak as usize) / (std::i16::MAX as usize);
            let top_row = (centre_row - spread.min(centre_row)) + 1;
            let bottom_row = (centre_row + spread).min(*height - 1);

            match spread {
                0 => {
                    arr[centre_row][i] = '-';
                },
                _ => {
                    for j in top_row..bottom_row {
                        arr[j][i] = 'o';
                    }
                }
            }
        });

        arr
    }

    pub fn get_samples(&mut self, start: &f64, end: &f64, num_bins: usize) -> Vec<f32> {
        let mut reader = WavReader::open(self.file.as_str()).unwrap();

        let clipped_start = (*start).min(1.).max(0.);
        let clipped_end = (*end).min(1.).max(0.);
        let all_samples = reader.samples::<i32>();
        let full_len = all_samples.len() as f64;
        let start_frame = ((full_len - 1.) * clipped_start) as usize;
        let end_frame = ((full_len - 1.) * clipped_end) as usize;
        let num_frames = end_frame - start_frame;
        

        if num_frames == 0 {
            return vec![0f32; num_bins]
        }

        let skip = ((*end - *start) * full_len) / num_bins as f64;
        let interp_start = *start * full_len;
        //println!("{} {} {} {} {} {} {} {}", clipped_start, clipped_end, all_samples.len(), start_frame, end_frame, num_frames, skip, interp_start);
        let section: Vec<i32> = all_samples
            .skip(start_frame)
            .take(num_frames)
            .map(|s| s.unwrap())
            .collect();

        (0..num_bins).map(|n| {
            let interp_index = (interp_start + (skip * n as f64)) - start_frame as f64;
            //println!("{} {}", n, interp_index);
            match interp_index {
                ii if ii < 0f64 => 0f32,
                ii if ii >= num_frames as f64 => 0f32,
                _ => {
                    let int_index = interp_index as usize;
                    section[int_index] as f32
                }
            }

        }).collect::<Vec<f32>>()
    }

    pub fn draw_samples(&mut self, samples: Vec<f32>, width: &usize, height: &usize) -> Vec<Vec<char>> {
        let mut arr = vec![vec![' '; *width]; *height];
        samples.iter().enumerate().for_each(|(i, sample)| {
            let norm_sample = (sample / (2f32 * std::i16::MAX as f32)) + 0.5;
            let col = (norm_sample * *height as f32) as usize;
            //println!("draw_samples {} {} {} {}", i, sample, norm_sample, col);
            arr[col][i] = 'o';
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
        c.load("/Users/richard/Developer/wavr/resources/duskwolf.wav".to_string());
        let p = c.get_peaks(0., 1., 20);
        assert_eq!(p, [0f32, 1f32, 2f32, 3f32, 4f32, 5f32, 6f32, 7f32, 8f32, 9f32]);
    }
    
    #[test]
    fn draws_wave() {
        let mut c = Core::new();
        c.load("/Users/richard/Developer/wavr/resources/duskwolf.wav".to_string());
        let p = c.get_peaks(0., 1., 120);
        let w = c.draw_wave(p, &120, &30);
        w.iter().for_each(|row: &Vec<char>| println!("{:?}", row.iter().collect::<String>())); 
    }
    
    #[test]
    fn gets_samples() {
        let mut c = Core::new();
        c.load("/Users/richard/Developer/wavr/resources/duskwolf.wav".to_string());
        let s = c.get_samples(&0.3, &0.3015, 30);
        println!("{:?}", s);
        assert_eq!(s.len(), 30);
    }
    
    #[test]
    fn gets_samples_2() {
        let mut c = Core::new();
        c.load("/Users/richard/Developer/wavr/resources/duskwolf.wav".to_string());
        let s = c.get_samples(&0.3, &0.3015, 164);
        println!("{:?}", s);
        assert_eq!(s.len(), 164);
    }
    
    #[test]
    fn draws_samples_big() {
        let mut c = Core::new();
        c.load("/Users/richard/Developer/wavr/resources/duskwolf.wav".to_string());
        let s = c.get_samples(&0.1, &0.9, 120);
        let w = c.draw_samples(s, &120, &60);
    }

    #[test]
    fn draws_samples() {
        let mut c = Core::new();
        c.load("/Users/richard/Developer/wavr/resources/duskwolf.wav".to_string());
        let s = c.get_samples(&0.3, &0.3015, 30);
        let w = c.draw_samples(s, &120, &30);
        assert_eq!(w.len(), 30);
        assert_eq!(w[0].len(), 120);
    }

    #[test]
    fn out_of_bounds_peaks_are_zero() {
        let mut c = Core::new();
        c.load("/Users/richard/Developer/wavr/resources/duskwolf.wav".to_string());
        let p = c.get_peaks(-1., 0., 5);
        assert_eq!(p, [0f32, 0f32, 0f32, 0f32, 0f32]);
        let p = c.get_peaks(1., 2., 5);
        assert_eq!(p, [0f32, 0f32, 0f32, 0f32, 0f32]);
    }

    #[test]
    fn panic1() {
        let mut c = Core::new();
        c.load("/Users/richard/Developer/wavr/resources/duskwolf.wav".to_string());
        let p = c.get_peaks(0.074016, 0.401696, 181);
        let w = c.draw_wave(p, &181, &30);
        assert_eq!(w.len(), 30);
    }

    #[test]
    fn out_of_bounds_samples_are_zero() {
        let mut c = Core::new();
        c.load("/Users/richard/Developer/wavr/resources/duskwolf.wav".to_string());
        let p = c.get_samples(&-1., &0., 5);
        assert_eq!(p, [0f32, 0f32, 0f32, 0f32, 0f32]);
        let p = c.get_samples(&1., &2., 5);
        assert_eq!(p, [0f32, 0f32, 0f32, 0f32, 0f32]);
        let p = c.get_samples(&0.5, &1.5, 5);
        assert_eq!(p, [-161.0f32, -1867.0f32, 19.0f32, 0f32, 0f32]);
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
}
