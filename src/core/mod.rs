extern crate hound;
mod waveform;

use std::string::String;
use self::waveform::WaveForm;
use std;

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
        let max_points = self.summary.as_ref().unwrap().summary_1k.len() - 1;
        let skip = (end - start) / (num_peaks as f32);
        (0..num_peaks).map(|x| {
            let phase = start + (x as f32 * skip);
            match phase {
                p if p < 0f32 => 0f32,
                p if p >= 1f32 => 0f32,
                _ => {
                    let int_phase = (phase * max_points as f32) as usize;
                    //println!("{} {} {}", phase, int_phase, skip);
                    self.summary.as_ref().unwrap().summary_1k[int_phase]
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
    fn out_of_bounds_peaks_are_zero() {
        let mut c = Core::new();
        c.load("/Users/richard/Developer/wavr/resources/duskwolf.wav".to_string());
        let p = c.get_peaks(-1., 0., 5);
        assert_eq!(p, [0f32, 0f32, 0f32, 0f32, 0f32]);
        let p = c.get_peaks(1., 2., 5);
        assert_eq!(p, [0f32, 0f32, 0f32, 0f32, 0f32]);
    }
}
