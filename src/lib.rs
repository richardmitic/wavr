extern crate termion;
extern crate clap;

use termion::raw::{IntoRawMode, RawTerminal};
use termion::screen::*;
use termion::input::TermRead;
use termion::event::Key;
use termion::{clear, cursor, style};
use clap::{App};

use std::io::{BufWriter, Write, stdin, Stdin};
use std::{io, thread};
use std::sync::mpsc::{channel, Sender, Receiver};
use std::string::String;
use std::vec::Vec;

mod core;
use core::Core;

/// The upper and lower boundary char.
const HORZ_BOUNDARY: &'static str = "─";
/// The left and right boundary char.
const VERT_BOUNDARY: &'static str = "│";

/// The top-left corner
const TOP_LEFT_CORNER: &'static str = "┌";
/// The top-right corner
const TOP_RIGHT_CORNER: &'static str = "┐";
/// The bottom-left corner
const BOTTOM_LEFT_CORNER: &'static str = "└";
/// The bottom-right corner
const BOTTOM_RIGHT_CORNER: &'static str = "┘";


pub struct WavrApp {
    w: u16,
    h: u16,
    out: AlternateScreen<RawTerminal<BufWriter<io::Stdout>>>,
    core: core::Core
}

impl WavrApp {

    fn clear(&mut self) {
        write!(self.out, "{}{}{}", clear::All, style::Reset, cursor::Goto(1, 1)).unwrap();
    }

    fn draw_top_line(&mut self) {
        write!(self.out, "{}{}", cursor::Goto(1, 1), TOP_LEFT_CORNER).unwrap();
        for _ in 0 .. self.w {
            write!(self.out, "{}", HORZ_BOUNDARY).unwrap();
        }
        write!(self.out, "{}", TOP_RIGHT_CORNER).unwrap();
    }

    fn draw_bottom_line(&mut self) {
        write!(self.out, "{}{}", cursor::Goto(1, self.h + 2), BOTTOM_LEFT_CORNER).unwrap();
        for _ in 0 .. self.w {
            write!(self.out, "{}", HORZ_BOUNDARY).unwrap();
        }
        write!(self.out, "{}", BOTTOM_RIGHT_CORNER).unwrap();
    }

    fn draw_box_sides(&mut self) {
        for j in 2 .. self.h + 3 {
            write!(self.out, "{}{}{}{}",
                cursor::Goto(1, j), VERT_BOUNDARY,
                cursor::Goto(self.w + 2, j), VERT_BOUNDARY
            ).unwrap();
        }
    }

    fn draw_box(&mut self) {
        self.draw_top_line();
        self.draw_box_sides();
        self.draw_bottom_line();
    }

    fn write(&mut self, val: String) {
        write!(self.out, "{}{}", cursor::Goto(3, 3), val).unwrap();
        self.flush();
    }

    fn write_i64(&mut self, val: i64) {
        write!(self.out, "{}{}", cursor::Goto(3, 3), val).unwrap();
        self.flush();
    }

    fn draw(&mut self) {
        self.clear();
        self.draw_box();
        self.flush();
    }

    fn load(&mut self, f: String) -> u32 {
        self.core.load(f)
    }

    fn flush(&mut self) {
        self.out.flush().unwrap();
    }

    fn exit(&mut self) {
        write!(self.out, "{}", termion::cursor::Show).unwrap();
        self.flush();
    }
}


fn key_input_thread(input: Stdin, key_tx: Sender<Option<String>>) {
    for c in input.keys() {
        match c.unwrap() {
            Key::Char(value) => key_tx.send(Some(value.to_string())).unwrap(),
            _ => {}
        }
    }
}


pub fn some_stuff() {

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
            .multiple(true)
            .help("Sets the level of verbosity"))
        .get_matches();

    let filepath = matches.value_of("INPUT");

    let mut screen = AlternateScreen::from(
        BufWriter::with_capacity(1 << 14, io::stdout())
            .into_raw_mode()
            .unwrap()
    );
    write!(screen, "{}", termion::cursor::Hide).unwrap();
    screen.flush().unwrap();

    let termsize = termion::terminal_size().ok();
    let termwidth = termsize.map(|(w,_)| w - 2);
    let termheight = termsize.map(|(_,h)| h - 2);

    let mut app = WavrApp {
        w: termwidth.unwrap_or(30),
        h: termheight.unwrap_or(10),
        out: screen,
        core: Core::new(),
    };

    app.draw();

    if filepath.is_some() {
        let length:u32 = app.load(String::from(filepath.unwrap()));
        app.write(length.to_string());
    } else {
        return;
    }

    // keys
    let input = stdin();
    let (tx, rx) : (Sender<Option<String>>, Receiver<Option<String>>) = channel();
    let key_tx = tx.clone();

    let _key_thread = thread::spawn(move || key_input_thread(input, key_tx));

    for _n in 0..20 {
        let key : Option<String> = rx.recv().unwrap();
        let stop = key.map(|cmd_s| {
            match cmd_s.as_ref() {
                "q" => {
                    return true;
                },
                "p" => {
                    let samples = app.core.get_samples(1000, 1020, 20);
                    let text: Vec<String> = samples.map(|s| {s.unwrap().to_string()} ).collect();
                    app.write(text.join(" "));
                },
                _ => {
                    app.write(cmd_s.repeat(10));
                }
            }
            false
        });

        if stop.unwrap_or(false) {
            break;
        }
    }

    // let _result = key_thread.join();

    app.exit();
}
