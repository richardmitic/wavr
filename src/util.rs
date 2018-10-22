use std::path::Path;
use std::ffi::OsStr;

pub enum FileType {
    WAV,
    PCM
}

pub fn floor_n(x: usize, n: usize) -> usize {
    (x / n) * n
}

pub fn get_extension_from_filename(filename: &str) -> Option<&str> {
    Path::new(filename)
        .extension()
        .and_then(OsStr::to_str)
}

pub fn get_type(filename: &str) -> FileType {
    match get_extension_from_filename(filename) {
        Some("wav") => FileType::WAV,
        _ => FileType::PCM
    }
}
