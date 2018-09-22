extern crate clap;
extern crate termion;

mod core;

use clap::App;
use core::Core;
use std::io::{Write, stdin, stdout};
use termion::event::{Key};
use termion::input::TermRead;
use termion::raw::{IntoRawMode};
use termion::terminal_size;


fn print_wave_raw(core: &mut Core, width: &usize, height: &usize, start: &mut f32, end: &mut f32, screen: &mut Write) {
    let peaks = core.get_peaks(*start, *end, *width as u32);
    let wave = core.draw_wave(peaks, &width, &height);
    print_pixels_raw(wave, screen);
}

fn print_pixels_raw(wave: Vec<Vec<char>>, screen: &mut Write) {
    write!(screen, "{}", termion::cursor::Goto(1,1)).unwrap();
    wave.iter().enumerate().for_each(|(i, line)| {
        write!(screen, "{}", termion::cursor::Goto(1, (i + 1) as u16)).unwrap();
        for pixel in line {
            write!(screen, "{}", pixel).unwrap();
        }
    });
    write!(screen, "{}", termion::cursor::Goto(1,1)).unwrap();
    screen.flush().unwrap();
}

fn shift(begin: &mut f32, end: &mut f32, delta: &f32) {
    *begin += delta;
    *end += delta;
}

fn setup_screen(screen: &mut Write) {
    write!(screen, "{}{}", termion::clear::All, termion::cursor::Hide).unwrap();
    screen.flush().unwrap()
}

fn reset_screen(screen: &mut Write) {
    write!(screen, "{}{}", termion::clear::All, termion::cursor::Show).unwrap();
    screen.flush().unwrap()
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
        .arg(clap::Arg::with_name("width")
            .short("w")
            .default_value("")
            .help("Width of drawing"))
        .arg(clap::Arg::with_name("height")
            .short("h")
            .default_value("")
            .help("Height of drawing"))
        .arg(clap::Arg::with_name("shift")
            .short("s")
            .default_value("0.1")
            .help("Amount to shift left and right"))
        .get_matches();

    let size = terminal_size().unwrap();

    let filepath = matches.value_of("INPUT");
    let mut beginning = matches.value_of("begin").unwrap().parse().unwrap();
    let mut end = matches.value_of("end").unwrap().parse().unwrap();
    let width = matches.value_of("width").unwrap().parse().unwrap_or(size.0) as usize;
    let height = matches.value_of("height").unwrap().parse().unwrap_or(size.1) as usize;
    let shift_delta: f32 = matches.value_of("shift").unwrap().parse().unwrap();
    
    println!("{:?}", filepath);

    let mut c = Core::new();
    c.load(filepath.unwrap().to_string());

    //print_wave(&mut c, &width, &height, &mut beginning, &mut end);

    let stdin = stdin();
    let mut out = stdout().into_raw_mode().unwrap();

    setup_screen(&mut out);

    print_wave_raw(&mut c, &width, &height, &mut beginning, &mut end, &mut out);

    for key in stdin.keys() {
        match key.unwrap() {
            Key::Char('q') => break,
            Key::Left => {
                shift(&mut beginning, &mut end, &shift_delta);
                print_wave_raw(&mut c, &width, &height, &mut beginning, &mut end, &mut out);
            },
            Key::Right => {
                shift(&mut beginning, &mut end, &(-shift_delta));
                print_wave_raw(&mut c, &width, &height, &mut beginning, &mut end, &mut out);
            },
            _ => {}
        }
    }

    reset_screen(&mut out);
}
