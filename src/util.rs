use std::ffi::OsStr;
use std::path::Path;

pub enum FileType {
    WAV,
    FLAC,
    PCM,
}

pub fn get_extension_from_filename(filename: &str) -> Option<&str> {
    Path::new(filename).extension().and_then(OsStr::to_str)
}

pub fn get_type(filename: &str) -> FileType {
    match get_extension_from_filename(filename) {
        Some("wav") => FileType::WAV,
        Some("flac") => FileType::FLAC,
        _ => FileType::PCM,
    }
}
