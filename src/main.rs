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

/// A view into a wave. Beginning and end as numbers between 0. and 1.
type View = (f64, f64);

struct ViewPoint {
    begin: f64,
    end: f64,
    shift_delta: f64,
    zoom_delta: f64
}

impl ViewPoint {
    fn shift_right(&mut self) -> View {
        self.begin += self.shift_delta;
        self.end += self.shift_delta;
        (self.begin, self.end)
    }
    
    fn shift_left(&mut self) -> View {
        self.begin -= self.shift_delta;
        self.end -= self.shift_delta;
        (self.begin, self.end)
    }

    fn zoom_in(&mut self) -> View {
        let x = (self.end - self.begin) * self.zoom_delta; 
        self.begin += x;
        self.end -= x;
        (self.begin, self.end)
    }
    
    fn zoom_out(&mut self) -> View {
        let x = (self.end - self.begin) * self.zoom_delta; 
        self.begin -= x;
        self.end += x;
        (self.begin, self.end)
    }

    fn get_view(&mut self) -> View {
        (self.begin, self.end)
    }
}


fn print_wave_raw(core: &mut Core, width: &usize, height: &usize, view: View, screen: &mut Write) {
    if core.should_draw_samples(&(view.0 as f64), &(view.1 as f64)) {
        let samples = core.get_samples(&(view.0 as f64), &(view.1 as f64), *width as usize);
        let wave = core.draw_samples(samples, &width, &height);
        print_pixels_raw(wave, screen);
    } else {
        let peaks = core.get_peaks(view.0 as f32, view.1 as f32, *width as u32);
        let wave = core.draw_wave(peaks, &width, &height);
        print_pixels_raw(wave, screen);
    }
}

fn print_wave_samples_raw(core: &mut Core, width: &usize, height: &usize, view: View, screen: &mut Write) {
    let samples = core.get_samples(&(view.0 as f64), &(view.1 as f64), *width as usize);
    let wave = core.draw_samples(samples, &width, &height);
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
            .help("Amount to shift left and right per key press"))
        .arg(clap::Arg::with_name("zoom")
            .short("z")
            .default_value("0.1")
            .help("Amount to zoom in and out per key press"))
        .get_matches();

    let size = terminal_size().unwrap();

    let filepath = matches.value_of("INPUT");
    let width = matches.value_of("width").unwrap().parse().unwrap_or(size.0) as usize;
    let height = matches.value_of("height").unwrap().parse().unwrap_or(size.1) as usize;
    
    let mut view_point = ViewPoint {
        begin: matches.value_of("begin").unwrap().parse().unwrap(),
        end: matches.value_of("end").unwrap().parse().unwrap(),
        shift_delta: matches.value_of("shift").unwrap().parse().unwrap(),
        zoom_delta: matches.value_of("zoom").unwrap().parse().unwrap()
    };

    println!("{:?}", filepath);

    let mut c = Core::new();
    c.load(filepath.unwrap().to_string());

    let stdin = stdin();
    let mut out = stdout().into_raw_mode().unwrap();

    setup_screen(&mut out);

    let view = view_point.get_view();
    print_wave_raw(&mut c, &width, &height, view, &mut out);

    for key in stdin.keys() {
        match key.unwrap() {
            Key::Char('q') => break,
            Key::Left => {
                let view = view_point.shift_left();
                print_wave_raw(&mut c, &width, &height, view, &mut out);
            },
            Key::Right => {
                let view = view_point.shift_right();
                print_wave_raw(&mut c, &width, &height, view, &mut out);
            },
            Key::Up => {
                let view = view_point.zoom_in();
                print_wave_raw(&mut c, &width, &height, view, &mut out);
            },
            Key::Down => {
                let view = view_point.zoom_out();
                print_wave_raw(&mut c, &width, &height, view, &mut out);
            },
            _ => {}
        }
    }

    reset_screen(&mut out);
}
