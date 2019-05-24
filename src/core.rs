extern crate hound;
extern crate rand;

use self::hound::WavReader;
#[allow(unused_imports)]
use self::rand::Rng;
use pcm::{get_duration, read_wavesection, WaveSamplesChannel};
use std;
use std::string::String;
use util::{get_type, FileType};
use waveform::{WaveForm, WaveSection};

type WavePeaksChannel = Vec<Option<WaveSection>>;

fn scale(x: f64, in_min: f64, in_max: f64, out_min: f64, out_max: f64) -> f64 {
    (((out_max - out_min) * (x - in_min)) / (in_max - in_min)) + out_min
}

fn scale_vec(arr: Vec<f64>, in_min: f64, in_max: f64, out_min: f64, out_max: f64) -> Vec<f64> {
    arr.into_iter()
        .map(|x| scale(x, in_min, in_max, out_min, out_max))
        .collect()
}

fn split_into_channels(samples: WaveSamplesChannel, n: usize) -> Vec<WaveSamplesChannel> {
    (0..n)
        .map(|chan| {
            samples
                .iter()
                .skip(chan)
                .step_by(n)
                .map(|s_ptr| *s_ptr)
                .collect::<WaveSamplesChannel>()
        })
        .collect::<Vec<WaveSamplesChannel>>()
}

fn linear_interp_lookup(
    samples: &Vec<WaveSamplesChannel>,
    channel: usize,
    position: f64,
) -> Option<f32> {
    match position {
        _ if position < 0. => None,
        _ if position >= samples[0].len() as f64 => None,
        p if position as usize == samples[0].len() - 1 => samples[channel][p as usize],
        _ => {
            let x = position.fract() as f32;
            let idx = position as usize;
            Some(
                (samples[channel][idx + 1].unwrap() * x)
                    + (samples[channel][idx].unwrap() * (1. - x)),
            )
        }
    }
}

fn interp_lookup(
    samples: Vec<WaveSamplesChannel>,
    start_frame: usize,
    indices: Vec<f64>,
) -> Vec<WaveSamplesChannel> {
    let mut new_samples: Vec<WaveSamplesChannel> = vec![vec![None; indices.len()]; samples.len()];

    for i in 0..indices.len() {
        for channel in 0..samples.len() {
            new_samples[channel][i] =
                linear_interp_lookup(&samples, channel, indices[i] - start_frame as f64);
        }
    }

    new_samples
}

fn draw_outline(arr: &mut Vec<Vec<char>>) {
    let height = arr.len();
    let width = arr[0].len();
    let j_max = height - 1;
    let i_max = width - 1;

    for i in 0..width {
        arr[0][i] = '─'
    }

    for j in 1..height - 1 {
        arr[j][0] = '│';
        arr[j][i_max] = '│';
    }

    for i in 0..width {
        arr[j_max][i] = '─'
    }

    arr[0][0] = '┌';
    arr[0][i_max] = '┐';
    arr[j_max][0] = '└';
    arr[j_max][i_max] = '┘';
}

pub struct DisplayChars {
    rms: char,
    peak: char,
    zero: char,
    none: char,
    sample_low: char,
    sample_mid: char,
    sample_high: char,
}

pub struct Core {
    file: String,
    filetype: FileType,
    summary: Option<WaveForm>,
    chars: DisplayChars,
}

impl Core {
    pub fn new() -> Core {
        Core {
            file: String::new(),
            filetype: FileType::PCM,
            summary: Option::None,
            chars: DisplayChars {
                rms: 'o',
                peak: '·',
                zero: '=',
                none: '-',
                sample_low: '․',
                sample_mid: '·',
                sample_high: '˙',
            },
        }
    }

    pub fn load(&mut self, f: String, channels: Option<u16>) {
        self.file = f;
        self.filetype = get_type(&self.file);
        self.summary = Some(WaveForm::from_file(&self.file, &self.filetype, channels));
    }

    pub fn channels(&mut self) -> usize {
        self.summary.as_ref().unwrap().channels as usize
    }

    pub fn get_peaks(&mut self, start: &f64, end: &f64, num_peaks: u32) -> Vec<WavePeaksChannel> {
        let best_summary = self
            .summary
            .as_ref()
            .unwrap()
            .get_best_summary(start, end, &num_peaks);
        let max_points = best_summary[0].len() - 1;
        let skip = (*end - *start) / (num_peaks as f64);

        best_summary
            .iter()
            .map(|summary_channel| {
                (0..num_peaks)
                    .map(|x| {
                        let phase = *start + (x as f64 * skip);
                        match phase {
                            p if p < 0f64 => None,
                            p if p >= 1f64 => None,
                            _ => {
                                let interp_index = phase * max_points as f64;
                                let int_index = interp_index as usize;
                                let coeff = interp_index - interp_index.floor();
                                let x = summary_channel[int_index].clone();
                                let y = summary_channel[int_index + 1].clone();
                                let diff = y - x;
                                Some(x + (diff * coeff as f32))
                            }
                        }
                    })
                    .collect::<WavePeaksChannel>()
            })
            .collect()
    }

    pub fn draw_peaks(
        &mut self,
        peaks: WavePeaksChannel,
        width: &usize,
        height: &usize,
    ) -> Vec<Vec<char>> {
        let mut arr = vec![vec![' '; *width]; *height];
        peaks.iter().enumerate().for_each(|(i, sect)| {
            let full_scale_max = (std::i16::MAX) as f64;
            let full_scale_min = (std::i16::MIN) as f64;
            let this_scale_max = (*height - 1) as f64;
            let centre_row = scale(0., full_scale_min, full_scale_max, this_scale_max, 0.) as usize
                + (*height % 2);
            match *sect {
                Some(ws) => {
                    let max_scaled = scale(
                        ws.max as f64,
                        full_scale_min,
                        full_scale_max,
                        this_scale_max,
                        0.,
                    )
                    .max(0f64)
                        + 0.5;
                    let min_scaled = scale(
                        ws.min as f64,
                        full_scale_min,
                        full_scale_max,
                        this_scale_max,
                        0.,
                    )
                    .min(this_scale_max)
                        + 0.5;
                    let max_rms_scaled = scale(
                        ws.rms as f64,
                        full_scale_min,
                        full_scale_max,
                        this_scale_max,
                        0.,
                    )
                    .max(0f64)
                        + 0.5;
                    let min_rms_scaled = scale(
                        -(ws.rms as f64),
                        full_scale_min,
                        full_scale_max,
                        this_scale_max,
                        0.,
                    )
                    .min(this_scale_max)
                        + 0.5;
                    //println!("{} {} {} {}", max_scaled, min_scaled, max_rms_scaled, min_rms_scaled);
                    let max_idx = max_scaled as usize;
                    let min_idx = min_scaled as usize;
                    let max_rms_idx = max_rms_scaled as usize;
                    let min_rms_idx = min_rms_scaled as usize;
                    match min_idx == max_idx {
                        true => arr[centre_row][i] = self.chars.zero,
                        false => {
                            for j in max_idx..min_idx {
                                arr[j][i] = self.chars.peak;
                            }

                            for j in max_rms_idx..min_rms_idx {
                                arr[j][i] = self.chars.rms;
                            }
                        }
                    }
                }
                None => arr[centre_row][i] = self.chars.none,
            }
        });

        draw_outline(&mut arr);

        arr
    }

    pub fn draw_peaks_multichannel(
        &mut self,
        peaks: Vec<WavePeaksChannel>,
        width: &usize,
        height: &usize,
    ) -> Vec<Vec<char>> {
        let heights = self.channel_heights(height);
        peaks
            .into_iter()
            .zip(heights.into_iter())
            .map(|(chan, h)| self.draw_peaks(chan, width, &h))
            .flatten()
            .collect()
    }

    pub fn get_samples_multichannel(
        &mut self,
        start: &f64,
        end: &f64,
        num_bins: usize,
    ) -> Vec<WaveSamplesChannel> {
        match &self.filetype {
            &FileType::WAV => self.get_samples_multichannel_wav(start, end, num_bins),
            _ => self.get_samples_multichannel_pcm(start, end, num_bins),
        }
    }

    pub fn get_samples_multichannel_pcm(
        &mut self,
        start: &f64,
        end: &f64,
        num_bins: usize,
    ) -> Vec<WaveSamplesChannel> {
        let channels = self.channels();
        let full_len = get_duration(&self.file, &channels) as f64;
        let bin_indices = (0..num_bins).map(|x| x as f64).collect();
        let interp_indices = scale_vec(
            bin_indices,
            0.,
            num_bins as f64 - 1.,
            *start * full_len,
            *end * (full_len - 1.),
        );
        let start_frame = interp_indices[0].min(full_len - 1.).max(0.) as usize;
        let end_frame = interp_indices[num_bins - 1].min(full_len - 1.).max(0.) as usize;

        let num_samples = ((end_frame - start_frame) + 1) * channels;
        if num_samples == 0 {
            return vec![vec![None; num_bins]; self.channels()];
        }

        let section = read_wavesection(&self.file, start_frame * &channels, Some(num_samples));
        let multichannel_samples = split_into_channels(section, self.channels());
        interp_lookup(multichannel_samples, start_frame, interp_indices)
    }

    pub fn get_samples_multichannel_wav(
        &mut self,
        start: &f64,
        end: &f64,
        num_bins: usize,
    ) -> Vec<WaveSamplesChannel> {
        let mut reader = WavReader::open(self.file.as_str()).unwrap();
        let _spec = reader.spec();

        let full_len = reader.duration() as f64;
        let bin_indices = (0..num_bins).map(|x| x as f64).collect();
        let interp_indices = scale_vec(
            bin_indices,
            0.,
            num_bins as f64 - 1.,
            *start * full_len,
            *end * (full_len - 1.),
        );
        let start_frame = interp_indices[0].min(full_len - 1.).max(0.) as usize;
        let end_frame = interp_indices[num_bins - 1].min(full_len - 1.).max(0.) as usize;

        let num_samples = ((end_frame - start_frame) + 1) * self.channels();
        if num_samples == 0 {
            return vec![vec![None; num_bins]; self.channels()];
        }

        let _pos = reader.seek(start_frame as u32);
        let section = reader
            .samples::<i16>()
            .take(num_samples)
            .map(|s| Some(s.unwrap() as f32))
            .collect::<WaveSamplesChannel>();
        assert_eq!(num_samples, section.len());

        let multichannel_samples = split_into_channels(section, self.channels());
        interp_lookup(multichannel_samples, start_frame, interp_indices)
    }

    pub fn draw_samples(
        &mut self,
        samples: WaveSamplesChannel,
        width: &usize,
        height: &usize,
    ) -> Vec<Vec<char>> {
        let mut arr = vec![vec![' '; *width]; *height];
        let full_scale_max = (std::i16::MAX) as f64;
        let full_scale_min = (std::i16::MIN) as f64;
        let this_scale_max = (*height - 1) as f64;
        samples
            .into_iter()
            .enumerate()
            .for_each(|(i, sample)| match sample {
                None => {
                    let col =
                        scale(0., full_scale_min, full_scale_max, this_scale_max, 0.) as usize;
                    arr[col][i] = self.chars.none;
                }
                Some(x) => {
                    let col = scale(x as f64, full_scale_min, full_scale_max, this_scale_max, 0.);
                    if col.fract() < 0.4 {
                        arr[col as usize][i] = self.chars.sample_high;
                    } else if col.fract() > 0.6 {
                        arr[col as usize][i] = self.chars.sample_low;
                    } else {
                        arr[col as usize][i] = self.chars.sample_mid;
                    }
                }
            });

        draw_outline(&mut arr);

        arr
    }

    pub fn should_draw_samples(&mut self, start: &f64, end: &f64, width: &usize) -> bool {
        let range = *end - *start;
        let num_peaks =
            self.summary
                .as_ref()
                .unwrap()
                .get_best_summary(start, end, &(*width as u32))[0]
                .len() as f64
                * range;
        num_peaks < (width / 2) as f64
    }

    pub fn draw_samples_multichannel(
        &mut self,
        samples: Vec<WaveSamplesChannel>,
        width: &usize,
        height: &usize,
    ) -> Vec<Vec<char>> {
        let heights = self.channel_heights(height);
        samples
            .into_iter()
            .zip(heights)
            .flat_map(|(chan, this_height)| {
                self.draw_samples(chan.to_vec(), width, &this_height)
                    .into_iter()
            })
            .collect::<Vec<Vec<char>>>()
    }

    fn channel_heights(&mut self, full_height: &usize) -> Vec<usize> {
        let float_height = *full_height as f32 / self.channels() as f32;
        (0..self.channels())
            .map(|n| (float_height * (n + 1) as f32) as usize - (float_height * n as f32) as usize)
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn loads_file() {
        let mut c = Core::new();
        c.load("./resources/duskwolf.wav".to_string(), None);
        let p = c.get_peaks(&0., &1., 20);
        assert_eq!(p.len(), 1);
        assert_eq!(p[0].len(), 20);
    }

    #[test]
    fn draws_peaks() {
        let mut c = Core::new();
        c.load("./resources/duskwolf.wav".to_string(), None);
        let p = c.get_peaks(&0., &1., 120);
        let w = c.draw_peaks_multichannel(p, &120, &30);
        w.iter()
            .for_each(|row: &Vec<char>| println!("{:?}", row.iter().collect::<String>()));
    }

    #[test]
    fn draws_peaks_stereo() {
        let mut c = Core::new();
        c.load("./resources/oktava.wav".to_string(), None);
        let p = c.get_peaks(&0., &1., 120);
        let w = c.draw_peaks_multichannel(p, &120, &30);
        w.iter()
            .for_each(|row: &Vec<char>| println!("{:?}", row.iter().collect::<String>()));
    }

    #[test]
    fn draws_samples() {
        let mut c = Core::new();
        c.load("./resources/duskwolf.wav".to_string(), None);
        let s = c.get_samples_multichannel(&0.3, &0.3015, 30);
        let w = c.draw_samples_multichannel(s, &120, &30);
        assert_eq!(w.len(), 30);
        assert_eq!(w[0].len(), 120);
    }

    #[test]
    fn out_of_bounds_peaks_are_zero() {
        let mut c = Core::new();
        c.load("./resources/duskwolf.wav".to_string(), None);
        let p = c.get_peaks(&-1., &0., 5);
        assert_eq!(p[0].iter().fold(true, |a, b| a && b.is_none()), true);
        let p = c.get_peaks(&1., &2., 5);
        assert_eq!(p[0].iter().fold(true, |a, b| a && b.is_none()), true);
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
        let samples: Vec<Option<f32>> = (0..8).into_iter().map(|i| Some(i as f32)).collect();
        let channels = split_into_channels(samples, 2);
        assert_eq!(
            channels[0],
            vec![Some(0f32), Some(2f32), Some(4f32), Some(6f32)]
        );
        assert_eq!(
            channels[1],
            vec![Some(1f32), Some(3f32), Some(5f32), Some(7f32)]
        );
    }

    #[test]
    fn mono_samples_reshaped() {
        let samples: Vec<Option<f32>> = (0..8).into_iter().map(|i| Some(i as f32)).collect();
        let channels = split_into_channels(samples.clone(), 1);
        assert_eq!(channels[0], samples);
    }

    #[test]
    fn gets_mono_samples() {
        let mut c = Core::new();
        c.load("./resources/sine.wav".to_string(), None);
        let p = c.get_samples_multichannel(&0., &1., 480);
        assert_eq!(p.len(), 1);
        assert_eq!(p[0].len(), 480);
    }

    #[test]
    fn gets_stereo_samples_0_1() {
        let mut c = Core::new();
        c.load("./resources/stereo_ramp.wav".to_string(), None);
        let p = c.get_samples_multichannel(&0., &1., 8);
        assert_eq!(
            p[0],
            vec![
                Some(0f32),
                Some(1f32),
                Some(2f32),
                Some(3f32),
                Some(4f32),
                Some(5f32),
                Some(6f32),
                Some(7f32)
            ]
        );
        assert_eq!(
            p[1],
            vec![
                Some(8f32),
                Some(9f32),
                Some(10f32),
                Some(11f32),
                Some(12f32),
                Some(13f32),
                Some(14f32),
                Some(15f32)
            ]
        );
    }

    #[test]
    fn gets_stereo_samples_0_05() {
        let mut c = Core::new();
        c.load("./resources/stereo_ramp.wav".to_string(), None);
        let p = c.get_samples_multichannel(&0., &0.5, 4);
        assert_close_enough(p[0][0].unwrap() as f64, 0., 0.000001);
        assert_close_enough(p[0][3].unwrap() as f64, 3., 0.000001);
        assert_close_enough(p[1][0].unwrap() as f64, 8., 0.000001);
        assert_close_enough(p[1][3].unwrap() as f64, 11., 0.000001);
    }

    #[test]
    fn gets_stereo_samples_05_1() {
        let mut c = Core::new();
        c.load("./resources/stereo_ramp.wav".to_string(), None);
        let p = c.get_samples_multichannel(&0.5, &1., 4);
        assert_eq!(p[0], vec![Some(4f32), Some(5f32), Some(6f32), Some(7f32)]);
        assert_eq!(
            p[1],
            vec![Some(12f32), Some(13f32), Some(14f32), Some(15f32)]
        );
    }

    #[test]
    fn gets_stereo_samples_025_075() {
        let mut c = Core::new();
        c.load("./resources/stereo_ramp.wav".to_string(), None);
        let p = c.get_samples_multichannel(&0.25, &0.75, 4);
        assert_close_enough(p[0][0].unwrap() as f64, 2., 0.000001);
        assert_close_enough(p[0][3].unwrap() as f64, 5., 0.000001);
        assert_close_enough(p[1][0].unwrap() as f64, 10., 0.000001);
        assert_close_enough(p[1][3].unwrap() as f64, 13., 0.000001);
    }

    #[test]
    fn gets_stereo_samples_neg1_0() {
        let mut c = Core::new();
        c.load("./resources/stereo_ramp.wav".to_string(), None);
        let p = c.get_samples_multichannel(&-1., &0., 4);
        assert_eq!(
            p,
            vec![
                vec![None, None, None, Some(0f32)],
                vec![None, None, None, Some(8f32)]
            ]
        );
    }

    #[test]
    fn gets_stereo_samples_neg05_05() {
        let mut c = Core::new();
        c.load("./resources/stereo_ramp.wav".to_string(), None);
        let p = c.get_samples_multichannel(&-0.5, &0.5, 8);
        assert_eq!(p[0][0..4], [None, None, None, None]);
        assert_eq!(p[1][0..4], [None, None, None, None]);
        let _ = p[0][4..8].into_iter().map(|x| assert!(x.is_some()));
        let _ = p[1][4..8].into_iter().map(|x| assert!(x.is_some()));
    }

    #[test]
    fn gets_stereo_samples_1_2() {
        let mut c = Core::new();
        c.load("./resources/stereo_ramp.wav".to_string(), None);
        let p = c.get_samples_multichannel(&1., &2., 4);
        assert_eq!(p, vec![vec![None; 4]; 2]);
    }

    #[test]
    fn gets_stereo_samples_0_1_long() {
        let mut c = Core::new();
        c.load("./resources/stereo_ramp.wav".to_string(), None);
        let p = c.get_samples_multichannel(&0., &1., 12);
        assert_eq!(p[0][0], Some(0.0));
        assert_eq!(p[0][11], Some(7.0));
        assert_eq!(p[1][0], Some(8.0));
        assert_eq!(p[1][11], Some(15.0));
    }

    #[test]
    fn draws_stereo_samples_even_height() {
        let mut c = Core::new();
        c.load("./resources/stereo.wav".to_string(), None);
        let p = c.get_samples_multichannel(&0., &0.25, 120);
        let w = c.draw_samples_multichannel(p, &120, &40);
        assert_eq!(w.len(), 40);
        assert_eq!(w[0].len(), 120);
        assert_eq!(w[0].len(), 120);
        w.iter()
            .for_each(|row: &Vec<char>| println!("{:?}", row.iter().collect::<String>()));
    }

    #[test]
    fn draws_stereo_samples_odd_height() {
        let mut c = Core::new();
        c.load("./resources/stereo.wav".to_string(), None);
        let p = c.get_samples_multichannel(&0., &0.25, 120);
        let w = c.draw_samples_multichannel(p, &120, &41);
        assert_eq!(w.len(), 41);
        assert_eq!(w[0].len(), 120);
        assert_eq!(w[0].len(), 120);
        w.iter()
            .for_each(|row: &Vec<char>| println!("{:?}", row.iter().collect::<String>()));
    }

    #[test]
    fn interp_sample() {
        let samples = vec![vec![Some(0f32), Some(1f32)]];
        assert_close_enough(
            linear_interp_lookup(&samples, 0, 0.00).unwrap() as f64,
            0.00,
            0.0000001,
        );
        assert_close_enough(
            linear_interp_lookup(&samples, 0, 0.25).unwrap() as f64,
            0.25,
            0.0000001,
        );
        assert_close_enough(
            linear_interp_lookup(&samples, 0, 0.50).unwrap() as f64,
            0.50,
            0.0000001,
        );
        assert_close_enough(
            linear_interp_lookup(&samples, 0, 0.75).unwrap() as f64,
            0.75,
            0.0000001,
        );
        assert_close_enough(
            linear_interp_lookup(&samples, 0, 1.00).unwrap() as f64,
            1.00,
            0.0000001,
        );
        assert_close_enough(
            linear_interp_lookup(&samples, 0, 1.01).unwrap() as f64,
            1.00,
            0.0000001,
        );
        assert_eq!(linear_interp_lookup(&samples, 0, -0.01), None);
        assert_eq!(linear_interp_lookup(&samples, 0, 2.), None);
    }

    #[test]
    fn chooses_samples() {
        let mut c = Core::new();
        c.load("./resources/stereo.wav".to_string(), None);
        assert_eq!(c.should_draw_samples(&0., &1., &40), true);
    }

    #[test]
    fn chooses_peaks() {
        let mut c = Core::new();
        c.load("./resources/stereo.wav".to_string(), None);
        assert_eq!(c.should_draw_samples(&0., &1., &20), false);
    }

    #[ignore]
    #[test]
    fn look_for_panics() {
        let iters = 1000;
        let min_start = -1f64;
        let max_start = 0.5f64;
        let min_len = 0.1f64;
        let max_len = 2f64;
        let min_width: usize = 10;
        let max_width: usize = 300;
        let min_height: usize = 10;
        let max_height: usize = 100;

        let mut rng = rand::thread_rng();

        for _ in 0..iters {
            let start = rng.gen_range::<f64>(min_start, max_start);
            let end = start + rng.gen_range::<f64>(min_len, max_len);
            let w = rng.gen_range::<usize>(min_width, max_width);
            let h = rng.gen_range::<usize>(min_height, max_height);
            println!("{} {} {} {}", start, end, w, h);

            let mut c = Core::new();
            c.load("./resources/stereo.wav".to_string(), None);
            let a = c.get_samples_multichannel(&start, &end, w);
            let _b = c.draw_samples_multichannel(a, &w, &h);
        }
    }
}
