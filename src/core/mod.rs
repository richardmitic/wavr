extern crate hound;
mod waveform;

use std::string::String;
use self::waveform::WaveForm;
use std;
use self::hound::WavReader;

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
            let top_row = centre_row - spread + 1;
            let bottom_row = centre_row + spread;

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
        let all_samples = reader.samples::<i32>();
        let start_frame = (all_samples.len() as f64 * *start) as usize;
        let end_frame = (all_samples.len() as f64 * *end) as usize;
        let num_frames = end_frame - start_frame;
        let section: Vec<i32> = all_samples
            .skip(start_frame)
            .take(num_frames)
            .map(|s| s.unwrap())
            .collect();

        let skip = ((num_frames - 1) as f32) / (num_bins as f32);
        (0..num_bins).map(|n| {
            let interp_index = n as f32 * skip;
            let int_index = interp_index as usize;
            let x = section[int_index] as f32;
            let y = section[int_index + 1] as f32;
            let diff = y - x;
            x + (diff * interp_index.fract())
        }).collect::<Vec<f32>>()
    }

    pub fn draw_samples(&mut self, samples: Vec<f32>, width: &usize, height: &usize) -> Vec<Vec<char>> {
        let mut arr = vec![vec![' '; *width]; *height];
        samples.iter().enumerate().for_each(|(i, sample)| {
            let norm_sample = (sample / (2f32 * std::i16::MAX as f32)) + 0.5;
            let col = (norm_sample * *height as f32) as usize;
            println!("{} {} {} {}", i, sample, norm_sample, col);
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
}
