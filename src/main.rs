#![feature(test)]
extern crate clap;
extern crate termion;

mod core;
mod pcm;
mod util;
mod waveform;
mod signals;

use clap::App;
use core::Core;
use std::io::{stdin, stdout, Write};
use termion::event::Key;
use termion::input::TermRead;
use termion::raw::IntoRawMode;
use termion::terminal_size;

const APP_VERSION: &'static str = env!("CARGO_PKG_VERSION");
const APP_NAME: &'static str = env!("CARGO_PKG_NAME");


/// A view into a wave. Beginning and end as numbers between 0. and 1.
type View = (f64, f64);

struct ViewPoint {
    begin: f64,
    end: f64,
    shift_delta: f64,
    zoom_delta: f64,
}

impl ViewPoint {
    fn shift_right(&mut self) -> View {
        let range = self.end - self.begin;
        self.begin += range * self.shift_delta;
        self.end += range * self.shift_delta;
        (self.begin, self.end)
    }

    fn shift_left(&mut self) -> View {
        let range = self.end - self.begin;
        self.begin -= range * self.shift_delta;
        self.end -= range * self.shift_delta;
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

    fn reset(&mut self) -> View {
        self.begin = 0.;
        self.end = 1.;
        (self.begin, self.end)
    }

    fn get_view(&mut self) -> View {
        (self.begin, self.end)
    }
}

fn print_wave(
    core: &mut Core,
    width: &usize,
    height: &usize,
    view: View,
    screen: &mut dyn Write,
    stdout: bool,
    draw_spect: bool,
) {
    if draw_spect {
        let spect = core.get_spect(&(view.0 as f64), &(view.1 as f64), *width as usize);
        let bitmap = core.draw_spect_multichannel(spect, &width, &height);
        if stdout {
            print_pixels_to_stdout(bitmap, screen)
        } else {
            print_pixels(bitmap, screen, view)
        }
    } else if core.should_draw_samples(&(view.0 as f64), &(view.1 as f64), width) {
        let samples =
            core.get_samples_multichannel(&(view.0 as f64), &(view.1 as f64), *width as usize);
        let wave = core.draw_samples_multichannel(samples, &width, &height);
        if stdout {
            print_pixels_to_stdout(wave, screen)
        } else {
            print_pixels(wave, screen, view)
        }
    } else {
        let peaks = core.get_peaks(&(view.0 as f64), &(view.1 as f64), *width as u32);
        let wave = core.draw_peaks_multichannel(peaks, &width, &height);
        if stdout {
            print_pixels_to_stdout(wave, screen);
        } else {
            print_pixels(wave, screen, view)
        }
    }
}

fn print_pixels_to_stdout(wave: Vec<Vec<char>>, screen: &mut dyn Write) {
    wave.iter().for_each(|line| {
        for pixel in line {
            write!(screen, "{}", pixel).unwrap();
        }
        write!(screen, "\n").unwrap();
    });
    screen.flush().unwrap();
}

fn print_pixels(wave: Vec<Vec<char>>, screen: &mut dyn Write, view: View) {
    write!(
        screen,
        "{}{}",
        termion::style::Bold,
        termion::cursor::Goto(1, 1)
    )
    .unwrap();
    wave.iter().enumerate().for_each(|(i, line)| {
        write!(screen, "{}", termion::cursor::Goto(1, (i + 1) as u16)).unwrap();
        for pixel in line {
            write!(screen, "{}", pixel).unwrap();
        }
    });
    write!(
        screen,
        "{}{}{:.6}:{:.6}",
        termion::style::Reset,
        termion::cursor::Goto(2, 1),
        view.0,
        view.1
    )
    .unwrap();
    screen.flush().unwrap();
}

fn setup_screen(screen: &mut dyn Write) {
    write!(screen, "{}{}", termion::clear::All, termion::cursor::Hide).unwrap();
    screen.flush().unwrap()
}

fn reset_screen(screen: &mut dyn Write) {
    write!(
        screen,
        "{}{}{}",
        termion::clear::All,
        termion::cursor::Show,
        termion::cursor::Goto(1, 1)
    )
    .unwrap();
    screen.flush().unwrap()
}

fn main() {
    // arguments
    let matches = App::new(APP_NAME)
        .version(APP_VERSION)
        .about("view wav files in the terminal")
        .arg(
            clap::Arg::with_name("INPUT")
                .help("Sets the input file to use")
                .required(true)
                .index(1),
        )
        .arg(
            clap::Arg::with_name("v")
                .short("v")
                .help("Sets the level of verbosity"),
        )
        .arg(
            clap::Arg::with_name("begin")
                .short("b")
                .default_value("0.")
                .help("Begining point within the wave"),
        )
        .arg(
            clap::Arg::with_name("end")
                .short("e")
                .default_value("1.")
                .help("End point within the wave"),
        )
        .arg(
            clap::Arg::with_name("width")
                .short("w")
                .default_value("")
                .help("Width of drawing"),
        )
        .arg(
            clap::Arg::with_name("height")
                .short("h")
                .default_value("")
                .help("Height of drawing"),
        )
        .arg(
            clap::Arg::with_name("shift")
                .short("s")
                .default_value("0.1")
                .help("Amount to shift left and right per key press"),
        )
        .arg(
            clap::Arg::with_name("zoom")
                .short("z")
                .default_value("0.1")
                .help("Amount to zoom in and out per key press"),
        )
        .arg(
            clap::Arg::with_name("channels")
                .short("c")
                .default_value("1")
                .help("Number of channels whean reading from a raw PCM file"),
        )
        .arg(
            clap::Arg::with_name("print")
                .short("p")
                .takes_value(false)
                .help("Print single view and exit"),
        )
        .arg(
            clap::Arg::with_name("spect")
                .short("x")
                .takes_value(false)
                .help("Display spectrogram"),
        )
        .arg(
            clap::Arg::with_name("signals")
                .short("S")
                .takes_value(false)
                .help("Use experimental \"signals\" backend"),
        )
        .get_matches();

    let size = terminal_size().unwrap();

    let filepath = matches.value_of("INPUT");
    let channels = matches.value_of("channels").unwrap().parse().unwrap_or(1) as u16;
    let width = matches.value_of("width").unwrap().parse().unwrap_or(size.0) as usize;
    let height = matches
        .value_of("height")
        .unwrap()
        .parse()
        .unwrap_or(size.1) as usize;
    let just_print = matches.occurrences_of("print") > 0;
    let draw_spect = matches.occurrences_of("spect") > 0;
    let use_signals = matches.occurrences_of("signals") > 0;

    let mut view_point = ViewPoint {
        begin: matches.value_of("begin").unwrap().parse().unwrap(),
        end: matches.value_of("end").unwrap().parse().unwrap(),
        shift_delta: matches.value_of("shift").unwrap().parse().unwrap(),
        zoom_delta: matches.value_of("zoom").unwrap().parse().unwrap(),
    };

    println!("{:?}", filepath);

    let mut c = Core::new();
    c.use_signals_backend(use_signals);
    c.load(filepath.unwrap().to_string(), Some(channels));

    let view = view_point.get_view();

    if just_print {
        print_wave(
            &mut c,
            &width,
            &height,
            view,
            &mut stdout(),
            true,
            draw_spect,
        );
        return;
    }

    let stdin = stdin();
    let mut out = stdout().into_raw_mode().unwrap();

    setup_screen(&mut out);

    print_wave(&mut c, &width, &height, view, &mut out, false, draw_spect);

    for key in stdin.keys() {
        match key.unwrap() {
            Key::Char('q') => break,
            Key::Char('r') => {
                let view = view_point.reset();
                print_wave(&mut c, &width, &height, view, &mut out, false, draw_spect);
            }
            Key::Left => {
                let view = view_point.shift_left();
                print_wave(&mut c, &width, &height, view, &mut out, false, draw_spect);
            }
            Key::Right => {
                let view = view_point.shift_right();
                print_wave(&mut c, &width, &height, view, &mut out, false, draw_spect);
            }
            Key::Up => {
                let view = view_point.zoom_in();
                print_wave(&mut c, &width, &height, view, &mut out, false, draw_spect);
            }
            Key::Down => {
                let view = view_point.zoom_out();
                print_wave(&mut c, &width, &height, view, &mut out, false, draw_spect);
            }
            _ => {}
        }
    }

    reset_screen(&mut out);
}
