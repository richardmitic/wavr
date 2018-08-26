extern crate hound;

use self::hound::{WavSamples, WavIntoSamples, WavReader};
use std::string::String;
use std::iter::Take;
use std::vec::Vec;
use std::io::BufReader;
use std::fs::File;


pub struct Core {
    file: String,
    channels: u16
}

impl Core {
    pub fn new() -> Core {
        Core {
            file: String::new(),
            channels: 1
        }
    }

    pub fn load(&mut self, f: String) -> u32 {
        self.file = f;
        let reader = hound::WavReader::open(&self.file).unwrap();
        self.channels = reader.spec().channels;

        reader.len() / (self.channels as u32)
    }

    pub fn get_samples(&mut self, start: u32, end: u32, num: u32) -> Take<WavIntoSamples<BufReader<File>, i16>> {
        let mut reader = WavReader::open(&self.file).unwrap();
        &reader.seek(start);
        reader.into_samples::<i16>().take((end - start) as usize)
    }
}
