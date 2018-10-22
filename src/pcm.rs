extern crate byteorder;

use std::cmp;
use std::io::SeekFrom;
use std::io::prelude::*;
use std::fs::{ File, metadata };
use self::byteorder::{ReadBytesExt, LittleEndian};
use std::mem::size_of;

pub type WaveSamplesChannel = Vec<Option<f32>>;


pub fn get_duration(path: &str, channels: &usize) -> usize {
    metadata(path).unwrap().len() as usize / size_of::<i16>() / channels
}

pub fn read_i16_section(path: &str, offset_samples: usize, num_samples: Option<usize>) -> Vec<i16> {
    let mut f = File::open(path).unwrap();
    let md = metadata(path).unwrap();
    let max_samples = (md.len() as usize / size_of::<i16>()) - offset_samples;

    let seek_bytes = offset_samples * size_of::<i16>();
    f.seek(SeekFrom::Start(seek_bytes as u64)).unwrap();

    let mut buffer = vec![0; cmp::min(num_samples.unwrap_or(max_samples), max_samples)];
    f.read_i16_into::<LittleEndian>(&mut buffer).unwrap();

    buffer.to_vec()
}

pub fn read_wavesection(path: &str, offset_samples: usize, num_samples: Option<usize>) -> WaveSamplesChannel {
    read_i16_section(path, offset_samples, num_samples).into_iter().map(|s| Some(s as f32)).collect::<WaveSamplesChannel>()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn read_pcm() {
        let samples = read_i16_section("./resources/stereo_ramp.pcm", 0, None);
        assert_eq!(samples.len(), 16);
    }

    #[test]
    fn read_pcm_section() {
        let samples = read_i16_section("./resources/stereo_ramp.pcm", 8, Some(8));
        assert_eq!(samples, vec![4i16, 12i16, 5i16, 13i16, 6i16, 14i16, 7i16, 15i16]);
    }
    
    #[test]
    fn read_too_much() {
        let samples = read_i16_section("./resources/stereo_ramp.pcm", 0, Some(17));
        assert_eq!(samples.len(), 16);
    }
}

