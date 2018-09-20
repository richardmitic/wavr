//extern crate libwavr;
extern crate clap;
extern crate termion;

mod core;

use clap::App;
use core::Core;
use std::io::{Write, stdin, stdout};
use termion::event::{Key};
use termion::input::TermRead;
use termion::raw::IntoRawMode;
use termion::terminal_size;

fn print_wave(core: &mut Core, width: &usize, height: &usize, start: &f32, end: &f32) {
    let peaks = core.get_peaks(*start, *end, *width as u32);
    let wave = core.draw_wave(peaks, &width, &height);
    print_pixels(wave);
}

fn print_pixels(wave: Vec<Vec<char>>) {
    print!("\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n");
    wave.iter().for_each(|row: &Vec<char>| println!("{:}", row.iter().collect::<String>()));
}

fn main() {

    // arguments
    let matches = App::new("WavR")
        .version("0.1")
        .about("view wav files in the terminal")
        .arg(clap::Arg::with_name("INPUT")
            .help("Sets the input file to use")
            .required(true)
            .index(1))
        .arg(clap::Arg::with_name("v")
            .short("v")
            .help("Sets the level of verbosity"))
        .arg(clap::Arg::with_name("begin")
            .short("b")
            .default_value("0.")
            .help("Begining point within the wave"))
        .arg(clap::Arg::with_name("end")
            .short("e")
            .default_value("1.")
            .help("End point within the wave"))
        .get_matches();

    let filepath = matches.value_of("INPUT");
    let begining = matches.value_of("begin").unwrap().parse().unwrap();
    let end = matches.value_of("end").unwrap().parse().unwrap();

    println!("{:?}", filepath);

    let mut c = Core::new();
    c.load(filepath.unwrap().to_string());

    let size = terminal_size().unwrap();
    let width = size.0 as usize;
    let height = size.1 as usize;

    print_wave(&mut c, &width, &height, &begining, &end);

    let stdin = stdin();
    let mut stdout = stdout().into_raw_mode().unwrap();

    for c in stdin.keys() {
        write!(stdout,
               "{}{}",
               termion::cursor::Goto(1, 1),
               termion::clear::CurrentLine)
                .unwrap();

        let event = c.unwrap();
        match event {
            Key::Char('q') => break,
            Key::Left => println!("left"),
            Key::Right => println!("right"),
            _ => {}
        }

        stdout.flush().unwrap();
    }

    write!(stdout, "{}", termion::cursor::Show).unwrap();
}
