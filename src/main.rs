//! Render ALL fractals >:3

#![feature(
	box_patterns,
	iter_intersperse,
)]

#![deny(
	unreachable_patterns,
	unsafe_code,
)]

use std::str::FromStr;

use clap::{Parser, arg};
use minifb::{Key, KeyRepeat, Window, WindowOptions};
use num::{complex::{Complex64, ComplexFloat}, BigUint, Zero};
use rand::{rng, Rng};
use rayon::iter::{IndexedParallelIterator, IntoParallelRefMutIterator, ParallelIterator};

mod utils_io;



const MOVE_STEP: float = 0.1;
const ZOOM_STEP: float = 1.1;

#[derive(Parser, Debug)]
#[clap(
	about,
	author,
	version,
	help_template = "\
		{before-help}{name} v{version}\n\
		\n\
		{about}\n\
		\n\
		Author: {author}\n\
		\n\
		{usage-heading} {usage}\n\
		\n\
		{all-args}{after-help}\
	",
)]
struct CliArgs {
	/// [optional] fractal Id or expression
	fractal: Option<String>,

	// TODO
	#[arg(short='s', long, default_value_t=false)]
	assign_zesc_once: bool,

	// TODO
	#[arg(short='b', long, default_value_t=false)]
	break_loop: bool,

	// TODO
	#[arg(short='e', long, default_value_t=100.)]
	zesc_value: float,

	// TODO
	#[arg(short='r', long, default_value_t=false)]
	keys_repeat: bool,

	// TODO
	#[arg(short='n', long, default_value=None)]
	get_id_of: Option<String>,

	// TODO: render fractal to imgs/vid
}

struct Params {
	fractal: (BigUint, Expr),
	assign_zesc_once: bool,
	break_loop: bool,
	zesc_value: float,
	keys_repeat: bool,
}
impl From<CliArgs> for Params {
	fn from(CliArgs {
		fractal,
		assign_zesc_once,
		break_loop,
		zesc_value,
		keys_repeat,
		get_id_of: _,
	}: CliArgs) -> Self {
		Self {
			fractal: {
				if let Some(fractal) = fractal {
					// TODO: base36
					if let Ok(id) = BigUint::from_str(&fractal) {
						let expr = Expr::from_int(&id);
						(id, expr)
					} else {
						let expr = Expr::from_str(&fractal).unwrap();
						let id = expr.to_int();
						(id, expr)
					}
				} else {
					let mut rng = rng();
					loop {
						const N_MAX_DIGITS: u32 = 19; // TODO?: support bigger numbers using `BigUint::from_str` and generate number of digits from poisson distribution
						let digits: u32 = rng.random_range(1 ..= N_MAX_DIGITS);
						let id: u64 = if digits == 1 {
							rng.random_range(0 ..= 9)
						} else {
							rng.random_range(10_u64.pow(digits-1) .. 10_u64.pow(digits))
						};
						let expr = Expr::from_u64(id);
						if expr.contains_z() {
							break (BigUint::from(id), expr);
						}
					}
				}
			},
			assign_zesc_once,
			break_loop,
			zesc_value,
			keys_repeat,
		}
	}
}



fn main() {
	let cli_args = CliArgs::parse();
	// dbg!(&cli_args);

	if let Some(expr_str) = cli_args.get_id_of {
		let expr = Expr::from_str(&expr_str).unwrap();
		let id = expr.to_int();
		println!("id: {id}");
		let expr_from_id = Expr::from_int(&id);
		assert_eq!(expr, expr_from_id, "expr from str != expr from int, which is bad...");
		println!("{} -> {}", id, expr_from_id.to_string());
		return
	}

	let mut params: Params = cli_args.into();
	println!("{} -> {}", params.fractal.0, params.fractal.1.to_string());

	let (mut w, mut h) = (320, 240);
	let (mut wf, mut hf) = (w as float, h as float);
	let mut buffer: Vec<u32> = vec![0; w * h];

	let mut window = Window::new(
		"fractals",
		w, h,
		WindowOptions {
			resize: true,
			// scale_mode: ScaleMode::Stretch,
			..WindowOptions::default()
		}
	).unwrap();

	window.set_target_fps(60);
	window.update_with_buffer(&buffer, w, h).unwrap();

	let mut zoom : float = 1.0;
	let mut cam_x: float = 0.0;
	let mut cam_y: float = 0.0;

	let mut quality = Quality::new();

	// let mut frame_i: u64 = 0;
	while window.is_open() && !window.is_key_down(Key::Escape) {
		let mut is_redraw_needed: bool = false;

		(w, h) = window.get_size();
		(wf, hf) = (w as float, h as float);
		let new_size = w * h;
		if new_size != buffer.len() {
			buffer.resize(new_size, 0);
			// println!("Resized to {w}x{h}");
			is_redraw_needed = true;
		}
		// let ratio_hw = hf / wf;
		// let ratio_wh = wf / hf;

		let move_speed = MOVE_STEP / zoom;
		if window.is_key_down(Key::Left)  || window.is_key_down(Key::H) || window.is_key_down(Key::A) {
			cam_x -= move_speed;
			is_redraw_needed = true;
		}
		if window.is_key_down(Key::Right) || window.is_key_down(Key::L) || window.is_key_down(Key::D) {
			cam_x += move_speed;
			is_redraw_needed = true;
		}
		if window.is_key_down(Key::Up)    || window.is_key_down(Key::K) || window.is_key_down(Key::W) {
			cam_y -= move_speed;
			is_redraw_needed = true;
		}
		if window.is_key_down(Key::Down)  || window.is_key_down(Key::J) || window.is_key_down(Key::S) {
			cam_y += move_speed;
			is_redraw_needed = true;
		}

		if window.is_key_pressed(Key::Space, KeyRepeat::No) {
			params.keys_repeat = !params.keys_repeat;
			println!("keys_repeat: {}", params.keys_repeat);
		}

		if window.is_key_pressed_or_down(Key::E, params.keys_repeat) {
			quality.increase();
			println!("quality: {:?}", quality);
			is_redraw_needed = true;
		}
		if window.is_key_pressed_or_down(Key::Q, params.keys_repeat) {
			quality.decrease();
			println!("quality: {:?}", quality);
			is_redraw_needed = true;
		}

		if window.is_key_pressed_or_down(Key::N, params.keys_repeat) {
			params.fractal.0 += 1_u32;
			params.fractal.1 = Expr::from_int(&params.fractal.0);
			println!("{} -> {}", params.fractal.0, params.fractal.1.to_string());
			is_redraw_needed = true;
		}
		if window.is_key_pressed_or_down(Key::P, params.keys_repeat) {
			if params.fractal.0 > 0_u32.into() {
				params.fractal.0 -= 1_u32;
				params.fractal.1 = Expr::from_int(&params.fractal.0);
				println!("{} -> {}", params.fractal.0, params.fractal.1.to_string());
				is_redraw_needed = true;
			}
		}

		if window.is_key_pressed_or_down(Key::B, params.keys_repeat) {
			params.break_loop = !params.break_loop;
			println!("break_loop: {}", params.break_loop);
			is_redraw_needed = true;
		}
		if window.is_key_pressed_or_down(Key::Y, params.keys_repeat) {
			params.assign_zesc_once = !params.assign_zesc_once;
			println!("assign_zesc_once: {}", params.assign_zesc_once);
			is_redraw_needed = true;
		}

		if window.is_key_pressed_or_down(Key::R, params.keys_repeat) {
			zoom  = 1.0;
			cam_x = 0.0;
			cam_y = 0.0;
			println!("zoom reset");
			is_redraw_needed = true;
		}

		// Compute center world coords BEFORE zoom
		let scx = w as float / 2.0; // screen center x
		let scy = h as float / 2.0; // screen center x
		let center_world_before = screen_to_world(STW{x:scx, y:scy, wf, hf, zoom, cam_x, cam_y});

		if window.is_key_down(Key::Z) || window.is_key_down(Key::I) {
			zoom *= ZOOM_STEP;
			is_redraw_needed = true;
		}
		if window.is_key_down(Key::X) || window.is_key_down(Key::O) {
			zoom /= ZOOM_STEP;
			is_redraw_needed = true;
		}

		if is_redraw_needed {
			// println!("\nframe {frame_i}:"); frame_i += 1;

			// println!("cam xy: {cam_x}, {cam_y}");
			// println!("zoom = {zoom}  ->  n_iters = {}", zoom_to_iters_n(zoom));
			// Compute center world coords AFTER zoom
			let center_world_after = screen_to_world(STW{x:scx, y:scy, wf, hf, zoom, cam_x, cam_y});
			// Adjust camera so center remains fixed
			cam_x += center_world_before.0 - center_world_after.0;
			cam_y += center_world_before.1 - center_world_after.1;

			buffer
				// .iter_mut()
				.par_iter_mut()
				.enumerate()
				.for_each(|(i, pixel)| {
					let x = (i % w) as float;
					let y = (i / w) as float;
					let (x, y) = screen_to_world(STW{x, y, wf, hf, zoom, cam_x, cam_y});
					let z_init = Complex64::new(x, y);
					let mut z = z_init;
					let mut z_prev = z_init;
					let mut z_last_not_nan = z;
					let mut z_esc = Complex64::zero();
					let mut is_bounded = true;
					let mut escape_iter_n: u32 = 0;
					let n_iters: u32 = quality.zoom_to_iters_n(zoom);
					for j in 0..n_iters {
						if !z.is_nan() {
							z_last_not_nan = z;
						}
						let z_new = params.fractal.1.eval(z, z_prev, z_init);
						z_prev = z;
						z = z_new;
						let z_check = z.norm() > params.zesc_value || z.is_nan();
						let is_bounded_check = if params.assign_zesc_once { is_bounded } else { true };
						if z_check && is_bounded_check {
							is_bounded = false;
							escape_iter_n = j;
							z_esc = z;
							if params.break_loop {
								break;
							}
						}
					}
					let color = if is_bounded { BLACK } else {
						// let t = (escape_iter_n as float) + 1. - z.abs().ln().ln() / (2.).ln();
						// let t = ((escape_iter_n + 1) as float) - z.abs().ln().log2();
						// println!("\nz_esc={z_esc}, escape_iter_n={escape_iter_n}, z={z}");
						assert!(!z_last_not_nan.is_nan(), "{z_last_not_nan}");
						let t = if !z_esc.is_nan() { z_esc } else { z_last_not_nan }.abs();
						FloatToColor::Rainbow.eval(t)
					};
					*pixel = color.value();
				});
		}

		window.update_with_buffer(&buffer, w, h).unwrap();
	}
}


#[allow(non_camel_case_types)]
type float = f64;


// const I: Complex64 = Complex64::I;


#[derive(Debug, Clone, Copy)]
struct Color(u32);
impl Color {
	fn value(&self) -> u32 { self.0 }
}

const WHITE: Color = Color(0xffffff);
const BLACK: Color = Color(0x000000);

const RED    : Color = Color(0xff0000);
const ORANGE : Color = Color(0xff8800);
const YELLOW : Color = Color(0xffff00);
const YELLEN : Color = Color(0x88ff00);
const GREEN  : Color = Color(0x00ff00);
const GRYAN  : Color = Color(0x00ff88);
const CYAN   : Color = Color(0x00ffff);
const BLYUAN : Color = Color(0x0088ff);
const BLUE   : Color = Color(0x0000ff);
const BRED   : Color = Color(0x8800ff);
const MAGENTA: Color = Color(0xff00ff);
const REGENTA: Color = Color(0xff0088);



enum FloatToColor {
	Rainbow,
}
impl FloatToColor {
	fn eval(&self, t: float) -> Color {
		// let t = ((t.ln() / 1e3).abs().ln() / 1e2).tanh() as f32;
		// 710 = ln(f64::MAX)
		// println!("t in eval before scaling = {t}");
		let t = ((t.ln() / 710.) * 2. - 1.).tanh() as f32;
		// println!("t in eval after scaling = {t}");

		fn lerp(t: f32, p: Color, q: Color) -> Color {
			// TODO: try `be`, `le`
			let [pa, pr, pg, pb] = p.0.to_ne_bytes();
			let [qa, qr, qg, qb] = q.0.to_ne_bytes();
			Color(u32::from_ne_bytes([
				((pa as f32) * (1.-t) + (qa as f32) * t) as u8,
				((pr as f32) * (1.-t) + (qr as f32) * t) as u8,
				((pg as f32) * (1.-t) + (qg as f32) * t) as u8,
				((pb as f32) * (1.-t) + (qb as f32) * t) as u8,
			]))
		}

		/// assumes `points` are sorted by float
		fn multipoint_gradient_1d(t: f32, points: &[(f32, Color)]) -> Color {
			if t.is_nan() { return WHITE }
			// println!("t in mpg1 arg = {t}");
			let mut i = 1;
			// println!("t = {t}");
			while points[i].0 < t { i += 1; }
			// println!("i = {i}");
			let (u, color_prev) = points[i-1];
			let (v, color_next) = points[i];
			let t = (t - u) / (v - u);
			// println!("t in mpg1 after uv = {t}");
			// panic!();
			lerp(t, color_prev, color_next)
		}

		match self {
			Self::Rainbow => {
				multipoint_gradient_1d(t, &[
					(-1.01, WHITE),
					(( 0./11.)*2.-1., RED   ),
					(( 1./11.)*2.-1., ORANGE),
					(( 2./11.)*2.-1., YELLOW),
					(( 3./11.)*2.-1., YELLEN),
					(( 4./11.)*2.-1., GREEN ),
					(( 5./11.)*2.-1., GRYAN ),
					(( 6./11.)*2.-1., CYAN  ),
					(( 7./11.)*2.-1., BLYUAN),
					(( 8./11.)*2.-1., BLUE  ),
					(( 9./11.)*2.-1., BRED   ),
					((10./11.)*2.-1., MAGENTA),
					((11./11.)*2.-1., REGENTA),
					(1.01, WHITE),
				])
			}
		}
	}
}





#[derive(Debug)]
struct Quality(i32);
impl Quality {
	fn new() -> Self {
		Self(0)
	}

	fn zoom_to_iters_n(&self, zoom: float) -> u32 {
		let quality = self.0 as float;
		let log_base: float = 1. + (-quality / 5.).exp();
		20 + (zoom.log(log_base) as u32)
	}

	fn increase(&mut self) {
		self.0 += 1;
	}

	fn decrease(&mut self) {
		self.0 -= 1;
	}
}



struct STW {
	x: float,
	y: float,
	/// screen_w
	wf: float,
	/// screen_h
	hf: float,
	zoom: float,
	cam_x: float,
	cam_y: float,
}
fn screen_to_world(STW { x, y, wf, hf, zoom, cam_x, cam_y }: STW) -> (float, float) {
	let aspect_ratio = hf / wf;
	let world_x = cam_x + (x - wf * 0.5) / (wf * 0.5) / zoom;
	let world_y = cam_y + (y - hf * 0.5) / (hf * 0.5) / zoom * aspect_ratio;
	(world_x, world_y)
}



trait WindowExtIsKeyPressedOrDown {
	fn is_key_pressed_or_down(&self, key: Key, repeat: bool) -> bool;
}
impl WindowExtIsKeyPressedOrDown for Window {
	fn is_key_pressed_or_down(&self, key: Key, repeat: bool) -> bool {
		if repeat {
			self.is_key_down(key)
		} else {
			self.is_key_pressed(key, KeyRepeat::No)
		}
	}
}




/// Box::new(...)
fn bx<T>(value: T) -> Box<T> {
	Box::new(value)
}


fn cf(re: float, im: float) -> Complex64 {
	Complex64::new(re, im)
}
fn cfr(x: float) -> Complex64 {
	Complex64::new(x, 0.)
}
fn cfi(x: float) -> Complex64 {
	Complex64::new(0., x)
}



#[derive(Debug, Clone, PartialEq)]
enum Expr {
	// WARNING: IF CHANGED UPDATE CONSTS
	// 0 order
	// finite variants:
	Z,
	PrevZ,
	InitZ,
	I,
	// infinite variants:
	UInt(u64),
	Float(float),
	Complex(Complex64),
	// 1 order
	Neg(Box<Expr>),
	Abs(Box<Expr>), Arg(Box<Expr>),
	Re(Box<Expr>), Im(Box<Expr>), Conj(Box<Expr>),
	Exp(Box<Expr>), Ln(Box<Expr>),
	Sqrt(Box<Expr>),
	Sin(Box<Expr>), Cos(Box<Expr>), Tan(Box<Expr>),
	Sinh(Box<Expr>), Cosh(Box<Expr>), Tanh(Box<Expr>),
	Asin(Box<Expr>), Acos(Box<Expr>), Atan(Box<Expr>),
	Asinh(Box<Expr>), Acosh(Box<Expr>), Atanh(Box<Expr>),
	Round(Box<Expr>), Ceil(Box<Expr>), Floor(Box<Expr>),
	// 2 order
	Sum(Box<(Expr, Expr)>),
	Prod(Box<(Expr, Expr)>),
	Div(Box<(Expr, Expr)>),
	Pow(Box<(Expr, Expr)>),
	// WARNING: IF CHANGED UPDATE CONSTS

	// TODO: SqrtN?
}
impl Expr {
	// WARNING: UPDATE HERE IF ENUM CHANGED
	const NUMBER_OF_FINITE_VARIANTS: u64 = 4;
	const NUMBER_OF_INFINITE_VARIANTS: u64 = 3 + (10+8+6) + 4;

	fn eval(&self, z: Complex64, prev_z: Complex64, init_z: Complex64) -> Complex64 {
		use Expr::*;
		match &self {
			// 0 order
			Z     => z,
			PrevZ => prev_z,
			InitZ => init_z,
			I     => Complex64::I,
			UInt(n)    => cfr(*n as float),
			Float(x)   => cfr(*x),
			Complex(c) => *c,
			// 1 order
			Neg(e)   => -e.eval(z, prev_z, init_z),
			Abs(e)   => cfr(e.eval(z, prev_z, init_z).abs()),
			Arg(e)   => cfr(e.eval(z, prev_z, init_z).arg()),
			Re(e)    => e.eval(z, prev_z, init_z).re().into(),
			Im(e)    => e.eval(z, prev_z, init_z).im().into(),
			Conj(e)  => e.eval(z, prev_z, init_z).conj(),
			Exp(e)   => e.eval(z, prev_z, init_z).exp(),
			Ln(e)    => e.eval(z, prev_z, init_z).ln(),
			Sqrt(e)  => e.eval(z, prev_z, init_z).sqrt(),
			Sin(e)   => e.eval(z, prev_z, init_z).sin(),
			Cos(e)   => e.eval(z, prev_z, init_z).cos(),
			Tan(e)   => e.eval(z, prev_z, init_z).tan(),
			Sinh(e)  => e.eval(z, prev_z, init_z).sinh(),
			Cosh(e)  => e.eval(z, prev_z, init_z).cosh(),
			Tanh(e)  => e.eval(z, prev_z, init_z).tanh(),
			Asin(e)  => e.eval(z, prev_z, init_z).asin(),
			Acos(e)  => e.eval(z, prev_z, init_z).acos(),
			Atan(e)  => e.eval(z, prev_z, init_z).atan(),
			Asinh(e) => e.eval(z, prev_z, init_z).asinh(),
			Acosh(e) => e.eval(z, prev_z, init_z).acosh(),
			Atanh(e) => e.eval(z, prev_z, init_z).atanh(),
			Round(e) => { let Complex64 { re, im } = e.eval(z, prev_z, init_z); cf(re.round(), im.round()) },
			Ceil(e)  => { let Complex64 { re, im } = e.eval(z, prev_z, init_z); cf(re.ceil(), im.ceil()) },
			Floor(e) => { let Complex64 { re, im } = e.eval(z, prev_z, init_z); cf(re.floor(), im.floor()) },
			// 2 order
			Sum(box(l, r))       => l.eval(z, prev_z, init_z) + r.eval(z, prev_z, init_z), // es.into_iter().map(|e| e.eval(z, prev_z, init_z)).sum(),
			Prod(box(l, r))      => l.eval(z, prev_z, init_z) * r.eval(z, prev_z, init_z), // es.into_iter().map(|e| e.eval(z, prev_z, init_z)).product(),
			Div(box(num, denom)) => num.eval(z, prev_z, init_z) / denom.eval(z, prev_z, init_z),
			Pow(box(b, t))       => b.eval(z, prev_z, init_z).powc(t.eval(z, prev_z, init_z)),
			// _ => todo!()
		}
	}

	fn new_random() -> Self {
		let id = todo!();
		Self::from_int(id)
	}

	fn from_u64(id: u64) -> Self {
		Self::from_int(&BigUint::from(id))
	}

	/*
	example:

	0 - z
	1 - int
	2 - sin
	3 - sum

	split n = use "snake" algo -> k, l

	f n --> expr
	0    z
	1    int 0
	2    sin (f 0) = sin z
	3    sum (split 0) = sum(f 0, f 0) = sum z z
	4    int 1
	5    sin (f 1) = sin (int 0)
	6    sum (split 1) = sum(f 0, f 1) = sum z (int 0)
	7    int 2
	8    sin (f 2) = sin (sin z)
	9    sum (split 2) = sum(f 1, f 0) = sum (int 0) z
	10   int 3
	11   sin (f 3) = sin (sum z z)
	12   sum (split 3) = sum(f 0, f 2) = sum z (sin z)
	13   int 4
	14   sin (f 4) = sin 1
	15   sum (split 4) = sum(f 1, f 1) = sum (int 0) (int 0)
	16   int 5
	17   sin (f 5) = sin (sin (int 0))
	18   sum (split 5) = sum(f 2, f 0) = sum (sin z) z
	...
	*/
	fn from_int(id: &BigUint) -> Self {
		use Expr::*;
		// dbg!(&id);
		if *id < BigUint::from(Self::NUMBER_OF_FINITE_VARIANTS) {
			let id: u32 = *id.to_u32_digits().get(0).unwrap_or(&0);
			match id {
				0 => Z,
				1 => PrevZ,
				2 => InitZ,
				3 => I,
				_ => unreachable!()
			}
		}
		else {
			let id = id - Self::NUMBER_OF_FINITE_VARIANTS;
			let id_mod: u32 = *(&id % Self::NUMBER_OF_INFINITE_VARIANTS).to_u32_digits().get(0).unwrap_or(&0);
			let id_inner = &id / Self::NUMBER_OF_INFINITE_VARIANTS;
			let id_inner_u64: u64 = *id_inner.to_u64_digits().get(0).unwrap_or(&0);
			match id_mod {
				0 => UInt(id_inner_u64),
				1 => Float(float::from_bits(id_inner_u64)), // TODO: rewrite to use more common numbers?
				2 => {
					let (k, l) = snake_2d_split_u64(id_inner_u64);
					let re = f32::from_bits(k as u32) as float;
					let im = f32::from_bits(l as u32) as float;
					// TODO: rewrite to use more common numbers?
					Complex(Complex64::new(re, im))
				}
				// 1 order
				3 => Neg(bx(Self::from_int(&id_inner))),
				4 => Abs(bx(Self::from_int(&id_inner))),
				5 => Arg(bx(Self::from_int(&id_inner))),
				6 => Re(bx(Self::from_int(&id_inner))),
				7 => Im(bx(Self::from_int(&id_inner))),
				8 => Conj(bx(Self::from_int(&id_inner))),
				9 => Exp(bx(Self::from_int(&id_inner))),
				10 => Ln(bx(Self::from_int(&id_inner))),
				11 => Sqrt(bx(Self::from_int(&id_inner))),
				12 => Sin(bx(Self::from_int(&id_inner))),
				13 => Cos(bx(Self::from_int(&id_inner))),
				14 => Tan(bx(Self::from_int(&id_inner))),
				15 => Sinh(bx(Self::from_int(&id_inner))),
				16 => Cosh(bx(Self::from_int(&id_inner))),
				17 => Tanh(bx(Self::from_int(&id_inner))),
				18 => Asin(bx(Self::from_int(&id_inner))),
				19 => Acos(bx(Self::from_int(&id_inner))),
				20 => Atan(bx(Self::from_int(&id_inner))),
				21 => Asinh(bx(Self::from_int(&id_inner))),
				22 => Acosh(bx(Self::from_int(&id_inner))),
				23 => Atanh(bx(Self::from_int(&id_inner))),
				24 => Round(bx(Self::from_int(&id_inner))),
				25 => Ceil(bx(Self::from_int(&id_inner))),
				26 => Floor(bx(Self::from_int(&id_inner))),
				// 2 order
				27 => {
					let (k, l) = snake_2d_split(&id_inner);
					let u = Self::from_int(&k);
					let v = Self::from_int(&l);
					Sum(bx((u, v)))
				}
				28 => {
					let (k, l) = snake_2d_split(&id_inner);
					let u = Self::from_int(&k);
					let v = Self::from_int(&l);
					Prod(bx((u, v)))
				}
				29 => {
					let (k, l) = snake_2d_split(&id_inner);
					let u = Self::from_int(&k);
					let v = Self::from_int(&l);
					Div(bx((u, v)))
				}
				30 => {
					let (k, l) = snake_2d_split(&id_inner);
					let u = Self::from_int(&k);
					let v = Self::from_int(&l);
					Pow(bx((u, v)))
				}
				_ => unreachable!()
			}
		}
	}

	fn contains_z(&self) -> bool {
		use Expr::*;
		match self {
			Z     => true,
			PrevZ => false,
			InitZ => false,
			I     => false,
			UInt(_n)    => false,
			Float(_x)   => false,
			Complex(_c) => false,
			// 1 order
			Neg(e)   => e.contains_z(),
			Abs(e)   => e.contains_z(),
			Arg(e)   => e.contains_z(),
			Re(e)    => e.contains_z(),
			Im(e)    => e.contains_z(),
			Conj(e)  => e.contains_z(),
			Exp(e)   => e.contains_z(),
			Ln(e)    => e.contains_z(),
			Sqrt(e)  => e.contains_z(),
			Sin(e)   => e.contains_z(),
			Cos(e)   => e.contains_z(),
			Tan(e)   => e.contains_z(),
			Sinh(e)  => e.contains_z(),
			Cosh(e)  => e.contains_z(),
			Tanh(e)  => e.contains_z(),
			Asin(e)  => e.contains_z(),
			Acos(e)  => e.contains_z(),
			Atan(e)  => e.contains_z(),
			Asinh(e) => e.contains_z(),
			Acosh(e) => e.contains_z(),
			Atanh(e) => e.contains_z(),
			Round(e) => e.contains_z(),
			Ceil(e)  => e.contains_z(),
			Floor(e) => e.contains_z(),
			// 2 order
			Sum(box(l, r))       => l.contains_z() || r.contains_z(),
			Prod(box(l, r))      => l.contains_z() || r.contains_z(),
			Div(box(num, denom)) => num.contains_z() || denom.contains_z(),
			Pow(box(b, t))       => b.contains_z() || t.contains_z(),
		}
	}

	fn to_int(&self) -> BigUint {
		use Expr::*;
		match self {
			// 0 order
			Z     => return 0_u32.into(),
			PrevZ => return 1_u32.into(),
			InitZ => return 2_u32.into(),
			I     => return 3_u32.into(),
			_ => {}
		}
		BigUint::from(Self::NUMBER_OF_FINITE_VARIANTS) + match self {
			// 0 order
			Z
			| PrevZ
			| InitZ
			| I
			=> unreachable!(),
			UInt(n)    => BigUint::from(00_u32) + n * Self::NUMBER_OF_INFINITE_VARIANTS,
			Float(x)   => BigUint::from(01_u32) + x.to_bits() * Self::NUMBER_OF_INFINITE_VARIANTS,
			Complex(c) => {
				let Complex64 { re, im } = c;
				let k: BigUint = (*re as f32).to_bits().into();
				let l: BigUint = (*im as f32).to_bits().into();
				BigUint::from(02_u32) + snake_2d_unsplit(&k, &l) * Self::NUMBER_OF_INFINITE_VARIANTS
			}
			// 1 order
			Neg(e)   => BigUint::from(03_u32) + e.to_int() * Self::NUMBER_OF_INFINITE_VARIANTS,
			Abs(e)   => BigUint::from(04_u32) + e.to_int() * Self::NUMBER_OF_INFINITE_VARIANTS,
			Arg(e)   => BigUint::from(05_u32) + e.to_int() * Self::NUMBER_OF_INFINITE_VARIANTS,
			Re(e)    => BigUint::from(06_u32) + e.to_int() * Self::NUMBER_OF_INFINITE_VARIANTS,
			Im(e)    => BigUint::from(07_u32) + e.to_int() * Self::NUMBER_OF_INFINITE_VARIANTS,
			Conj(e)  => BigUint::from(08_u32) + e.to_int() * Self::NUMBER_OF_INFINITE_VARIANTS,
			Exp(e)   => BigUint::from(09_u32) + e.to_int() * Self::NUMBER_OF_INFINITE_VARIANTS,
			Ln(e)    => BigUint::from(10_u32) + e.to_int() * Self::NUMBER_OF_INFINITE_VARIANTS,
			Sqrt(e)  => BigUint::from(11_u32) + e.to_int() * Self::NUMBER_OF_INFINITE_VARIANTS,
			Sin(e)   => BigUint::from(12_u32) + e.to_int() * Self::NUMBER_OF_INFINITE_VARIANTS,
			Cos(e)   => BigUint::from(13_u32) + e.to_int() * Self::NUMBER_OF_INFINITE_VARIANTS,
			Tan(e)   => BigUint::from(14_u32) + e.to_int() * Self::NUMBER_OF_INFINITE_VARIANTS,
			Sinh(e)  => BigUint::from(15_u32) + e.to_int() * Self::NUMBER_OF_INFINITE_VARIANTS,
			Cosh(e)  => BigUint::from(16_u32) + e.to_int() * Self::NUMBER_OF_INFINITE_VARIANTS,
			Tanh(e)  => BigUint::from(17_u32) + e.to_int() * Self::NUMBER_OF_INFINITE_VARIANTS,
			Asin(e)  => BigUint::from(18_u32) + e.to_int() * Self::NUMBER_OF_INFINITE_VARIANTS,
			Acos(e)  => BigUint::from(19_u32) + e.to_int() * Self::NUMBER_OF_INFINITE_VARIANTS,
			Atan(e)  => BigUint::from(20_u32) + e.to_int() * Self::NUMBER_OF_INFINITE_VARIANTS,
			Asinh(e) => BigUint::from(21_u32) + e.to_int() * Self::NUMBER_OF_INFINITE_VARIANTS,
			Acosh(e) => BigUint::from(22_u32) + e.to_int() * Self::NUMBER_OF_INFINITE_VARIANTS,
			Atanh(e) => BigUint::from(23_u32) + e.to_int() * Self::NUMBER_OF_INFINITE_VARIANTS,
			Round(e) => BigUint::from(24_u32) + e.to_int() * Self::NUMBER_OF_INFINITE_VARIANTS,
			Ceil(e)  => BigUint::from(25_u32) + e.to_int() * Self::NUMBER_OF_INFINITE_VARIANTS,
			Floor(e) => BigUint::from(26_u32) + e.to_int() * Self::NUMBER_OF_INFINITE_VARIANTS,
			// 2 order
			Sum(box(l, r))       => BigUint::from(27_u32) + snake_2d_unsplit(&l.to_int(), &r.to_int()) * Self::NUMBER_OF_INFINITE_VARIANTS,
			Prod(box(l, r))      => BigUint::from(28_u32) + snake_2d_unsplit(&l.to_int(), &r.to_int()) * Self::NUMBER_OF_INFINITE_VARIANTS,
			Div(box(num, denom)) => BigUint::from(29_u32) + snake_2d_unsplit(&num.to_int(), &denom.to_int()) * Self::NUMBER_OF_INFINITE_VARIANTS,
			Pow(box(b, t))       => BigUint::from(30_u32) + snake_2d_unsplit(&b.to_int(), &t.to_int()) * Self::NUMBER_OF_INFINITE_VARIANTS,
			// _ => todo!()
		}
	}
}

impl ToString for Expr {
	fn to_string(&self) -> String {
		use Expr::*;
		match self {
			// 0 order
			Z     => format!("Z"),
			PrevZ => format!("PrevZ"),
			InitZ => format!("InitZ"),
			I     => format!("I"),
			UInt(n)    => format!("{n}"),
			Float(x)   => format!("{x:e}"),
			Complex(z) => format!("{z:e}"),
			// 1 order
			Neg(e)   => format!("-({})", e.to_string()),
			Abs(e)   => format!("Abs({})", e.to_string()),
			Arg(e)   => format!("Arg({})", e.to_string()),
			Re(e)    => format!("Re({})", e.to_string()),
			Im(e)    => format!("Im({})", e.to_string()),
			Conj(e)  => format!("Conj({})", e.to_string()),
			Exp(e)   => format!("Exp({})", e.to_string()),
			Ln(e)    => format!("Ln({})", e.to_string()),
			Sqrt(e)  => format!("Sqrt({})", e.to_string()),
			Sin(e)   => format!("Sin({})", e.to_string()),
			Cos(e)   => format!("Cos({})", e.to_string()),
			Tan(e)   => format!("Tan({})", e.to_string()),
			Sinh(e)  => format!("Sinh({})", e.to_string()),
			Cosh(e)  => format!("Cosh({})", e.to_string()),
			Tanh(e)  => format!("Tanh({})", e.to_string()),
			Asin(e)  => format!("Asin({})", e.to_string()),
			Acos(e)  => format!("Acos({})", e.to_string()),
			Atan(e)  => format!("Atan({})", e.to_string()),
			Asinh(e) => format!("Asinh({})", e.to_string()),
			Acosh(e) => format!("Acosh({})", e.to_string()),
			Atanh(e) => format!("Atanh({})", e.to_string()),
			Round(e) => format!("Round({})", e.to_string()),
			Ceil(e)  => format!("Ceil({})", e.to_string()),
			Floor(e) => format!("Floor({})", e.to_string()),
			// 2 order
			Sum(box(l, r))       => format!("({}+{})", l.to_string(), r.to_string()), // es.into_iter().map(|e| e.to_string()).intersperse(format!("+")).collect(),
			Prod(box(l, r))      => format!("({}*{})", l.to_string(), r.to_string()), // es.into_iter().map(|e| e.to_string()).intersperse(format!("*")).collect(),
			Div(box(num, denom)) => format!("({})/({})", num.to_string(), denom.to_string()),
			Pow(box(b, t))       => format!("({})^({})", b.to_string(), t.to_string()),
			// _ => todo!()
		}
	}
}

#[derive(Debug, PartialEq, Eq)]
enum ExprFromStrErr {
	BadBracketsSequence,
	BracketClosingBeforeOpeningAt { index: usize },
	BadExpression,
}
impl ToString for ExprFromStrErr {
	fn to_string(&self) -> String {
		match self {
			Self::BadBracketsSequence => format!("bad brackets sequence"),
			Self::BracketClosingBeforeOpeningAt { index } => format!("closing bracket before opening at index {index}"),
			Self::BadExpression => format!("bad expression"),
		}
	}
}
impl ExprFromStrErr {
	fn shift_index_by(mut self, delta: usize) -> Self {
		match &mut self {
			Self::BracketClosingBeforeOpeningAt { index } => {
				*index += delta;
			}
			_ => {}
		}
		self
	}
}

impl FromStr for Expr {
	type Err = ExprFromStrErr;
	fn from_str(s: &str) -> Result<Self, Self::Err> {
		use Expr::*;
		// dbg!(s);
		let s: &str = &s.trim().to_lowercase();
		// dbg!(s);
		if s.starts_with('(') && s.ends_with(')') {
			return Self::from_str(&s[1..s.len()-1])
		}
		match s {
			"z" => return Ok(Z),
			"zinit" | "initz" => return Ok(InitZ),
			"zprev" | "prevz" => return Ok(PrevZ),
			"i" => return Ok(I),
			_ => {}
		}
		if let Ok(n) = u64::from_str(s) { return Ok(UInt(n)) }
		if let Ok(x) = float::from_str(s) { return Ok(Float(x)) }
		if let Ok(z) = Complex64::from_str(s) { return Ok(Complex(z)) }

		let mut level: i32 = 0;
		let mut index_of_lb: Option<usize> = None;
		let mut index_of_rb: Option<usize> = None;
		let mut index_of_plus: Option<usize> = None;
		let mut index_of_minus: Option<usize> = None;
		let mut index_of_mul: Option<usize> = None;
		let mut index_of_div: Option<usize> = None;
		let mut index_of_pow: Option<usize> = None;
		for (i, c) in s.chars().enumerate() {
			match c {
				'(' => {
					if level == 0 && index_of_lb.is_none() {
						index_of_lb = Some(i);
					}
					level += 1;
				}
				')' => {
					if level == 0 && index_of_rb.is_none() {
						index_of_rb = Some(i);
					}
					level -= 1;
				}
				'+' if level == 0 && index_of_plus.is_none() => { index_of_plus = Some(i) }
				'-' if level == 0 && index_of_minus.is_none() && i > 0 => { index_of_minus = Some(i) }
				'*' if level == 0 && index_of_mul.is_none() => { index_of_mul = Some(i) }
				'/' if level == 0 && index_of_div.is_none() => { index_of_div = Some(i) }
				'^' if level == 0 && index_of_pow.is_none() => { index_of_pow = Some(i) }
				_ => {}
			}
			// dbg!(level);
			if !(level >= 0) { return Err(ExprFromStrErr::BracketClosingBeforeOpeningAt { index: i }) }
		}
		// dbg!(level);
		return if !(level == 0) { Err(ExprFromStrErr::BadBracketsSequence) }
		else if let Some(i) = index_of_plus {
			let l = Self::from_str(&s[..i])?;
			let r = Self::from_str(&s[i+1..])
				.map_err(|e| e.shift_index_by(i+1))?;
			Ok(Sum(bx((l, r))))
		}
		else if let Some(i) = index_of_minus {
			let l = Self::from_str(&s[..i])?;
			let r = Self::from_str(&s[i+1..])
				.map_err(|e| e.shift_index_by(i+1))?;
			Ok(Sum(bx((l, Neg(bx(r))))))
		}
		else if let Some(i) = index_of_mul {
			let l = Self::from_str(&s[..i])?;
			let r = Self::from_str(&s[i+1..])
				.map_err(|e| e.shift_index_by(i+1))?;
			Ok(Prod(bx((l, r))))
		}
		else if let Some(i) = index_of_div {
			let l = Self::from_str(&s[..i])?;
			let r = Self::from_str(&s[i+1..])
				.map_err(|e| e.shift_index_by(i+1))?;
			Ok(Div(bx((l, r))))
		}
		else if let Some(i) = index_of_pow {
			let l = Self::from_str(&s[..i])?;
			let r = Self::from_str(&s[i+1..])
				.map_err(|e| e.shift_index_by(i+1))?;
			Ok(Pow(bx((l, r))))
		}

		else if s.starts_with("-") {
			let inner = Self::from_str(&s[1..])
				.map_err(|e| e.shift_index_by(1))?;
			Ok(Neg(bx(inner)))
		}
		else if s.starts_with("abs") {
			let inner = Self::from_str(&s[4..s.len()-1])
				.map_err(|e| e.shift_index_by(4))?;
			Ok(Abs(bx(inner)))
		}
		else if s.starts_with("arg") {
			let inner = Self::from_str(&s[4..s.len()-1])
				.map_err(|e| e.shift_index_by(4))?;
			Ok(Arg(bx(inner)))
		}
		else if s.starts_with("re") {
			let inner = Self::from_str(&s[3..s.len()-1])
				.map_err(|e| e.shift_index_by(3))?;
			Ok(Re(bx(inner)))
		}
		else if s.starts_with("im") {
			let inner = Self::from_str(&s[3..s.len()-1])
				.map_err(|e| e.shift_index_by(3))?;
			Ok(Im(bx(inner)))
		}
		else if s.starts_with("conj") {
			let inner = Self::from_str(&s[5..s.len()-1])
				.map_err(|e| e.shift_index_by(5))?;
			Ok(Conj(bx(inner)))
		}
		else if s.starts_with("exp") {
			let inner = Self::from_str(&s[4..s.len()-1])
				.map_err(|e| e.shift_index_by(4))?;
			Ok(Exp(bx(inner)))
		}
		else if s.starts_with("ln") {
			let inner = Self::from_str(&s[3..s.len()-1])
				.map_err(|e| e.shift_index_by(3))?;
			Ok(Ln(bx(inner)))
		}
		else if s.starts_with("sqrt") {
			let inner = Self::from_str(&s[5..s.len()-1])
				.map_err(|e| e.shift_index_by(5))?;
			Ok(Sqrt(bx(inner)))
		}
		else if s.starts_with("sinh") {
			let inner = Self::from_str(&s[5..s.len()-1])
				.map_err(|e| e.shift_index_by(5))?;
			Ok(Sinh(bx(inner)))
		}
		else if s.starts_with("cosh") {
			let inner = Self::from_str(&s[5..s.len()-1])
				.map_err(|e| e.shift_index_by(5))?;
			Ok(Cosh(bx(inner)))
		}
		else if s.starts_with("tanh") {
			let inner = Self::from_str(&s[5..s.len()-1])
				.map_err(|e| e.shift_index_by(5))?;
			Ok(Tanh(bx(inner)))
		}
		else if s.starts_with("sin") {
			let inner = Self::from_str(&s[4..s.len()-1])
				.map_err(|e| e.shift_index_by(4))?;
			Ok(Sin(bx(inner)))
		}
		else if s.starts_with("cos") {
			let inner = Self::from_str(&s[4..s.len()-1])
				.map_err(|e| e.shift_index_by(4))?;
			Ok(Cos(bx(inner)))
		}
		else if s.starts_with("tan") {
			let inner = Self::from_str(&s[4..s.len()-1])
				.map_err(|e| e.shift_index_by(4))?;
			Ok(Tan(bx(inner)))
		}
		else if s.starts_with("asinh") {
			let inner = Self::from_str(&s[6..s.len()-1])
				.map_err(|e| e.shift_index_by(6))?;
			Ok(Asinh(bx(inner)))
		}
		else if s.starts_with("acosh") {
			let inner = Self::from_str(&s[6..s.len()-1])
				.map_err(|e| e.shift_index_by(6))?;
			Ok(Acosh(bx(inner)))
		}
		else if s.starts_with("atanh") {
			let inner = Self::from_str(&s[6..s.len()-1])
				.map_err(|e| e.shift_index_by(6))?;
			Ok(Atanh(bx(inner)))
		}
		else if s.starts_with("asin") {
			let inner = Self::from_str(&s[5..s.len()-1])
				.map_err(|e| e.shift_index_by(5))?;
			Ok(Asin(bx(inner)))
		}
		else if s.starts_with("acos") {
			let inner = Self::from_str(&s[5..s.len()-1])
				.map_err(|e| e.shift_index_by(5))?;
			Ok(Acos(bx(inner)))
		}
		else if s.starts_with("atan") {
			let inner = Self::from_str(&s[5..s.len()-1])
				.map_err(|e| e.shift_index_by(5))?;
			Ok(Atan(bx(inner)))
		}
		else if s.starts_with("round") {
			let inner = Self::from_str(&s[6..s.len()-1])
				.map_err(|e| e.shift_index_by(6))?;
			Ok(Round(bx(inner)))
		}
		else if s.starts_with("ceil") {
			let inner = Self::from_str(&s[5..s.len()-1])
				.map_err(|e| e.shift_index_by(5))?;
			Ok(Ceil(bx(inner)))
		}
		else if s.starts_with("floor") {
			let inner = Self::from_str(&s[6..s.len()-1])
				.map_err(|e| e.shift_index_by(6))?;
			Ok(Floor(bx(inner)))
		}
		else {
			Err(ExprFromStrErr::BadExpression)
		}
	}
}





fn snake_2d_split(n: &BigUint) -> (BigUint, BigUint) {
	if *n == BigUint::ZERO { return (BigUint::ZERO, BigUint::ZERO) }
	/* n -> (x, y)
	0  2  5  9  14 20 27 35
	1  4  8  13 19 26 34
	3  7  12 18 25 33
	6  11 17 24 32
	10 16 23 31
	15 22 30
	21 29
	28
	*/
	let row_i = ((1_u32 + 8_u32 * n).sqrt() - 1_u32) / 2_u32;
	let sum = &row_i * (&row_i + 1_u32) / 2_u32;
	// dbg!(&row_i, &sum);
	let delta = n - sum;
	let k = delta.clone();
	let l = row_i - delta;
	(k, l)
}

fn snake_2d_unsplit(k: &BigUint, l: &BigUint) -> BigUint {
	let row_i = k + l;
	let sum = &row_i * (&row_i + 1_u32) / 2_u32;
	let delta = k;
	let n = sum + delta;
	n
}

fn snake_2d_split_u64(n: u64) -> (u64, u64) {
	let (k, l) = snake_2d_split(&BigUint::from(n));
	let k = *k.to_u64_digits().get(0).unwrap_or(&0);
	let l = *l.to_u64_digits().get(0).unwrap_or(&0);
	(k, l)
}

fn snake_2d_unsplit_u64(k: u64, l: u64) -> u64 {
	*snake_2d_unsplit(&k.into(), &l.into()).to_u64_digits().get(0).unwrap_or(&0)
}





#[cfg(test)]
mod snake_2d {
	use super::*;

	mod split_u64 {
		use super::*;
		#[test] fn _00() { assert_eq!(snake_2d_split_u64(00), (0, 0)) }

		#[test] fn _01() { assert_eq!(snake_2d_split_u64(01), (0, 1)) }
		#[test] fn _02() { assert_eq!(snake_2d_split_u64(02), (1, 0)) }

		#[test] fn _03() { assert_eq!(snake_2d_split_u64(03), (0, 2)) }
		#[test] fn _04() { assert_eq!(snake_2d_split_u64(04), (1, 1)) }
		#[test] fn _05() { assert_eq!(snake_2d_split_u64(05), (2, 0)) }

		#[test] fn _06() { assert_eq!(snake_2d_split_u64(06), (0, 3)) }
		#[test] fn _07() { assert_eq!(snake_2d_split_u64(07), (1, 2)) }
		#[test] fn _08() { assert_eq!(snake_2d_split_u64(08), (2, 1)) }
		#[test] fn _09() { assert_eq!(snake_2d_split_u64(09), (3, 0)) }

		#[test] fn _10() { assert_eq!(snake_2d_split_u64(10), (0, 4)) }
		#[test] fn _11() { assert_eq!(snake_2d_split_u64(11), (1, 3)) }
		#[test] fn _12() { assert_eq!(snake_2d_split_u64(12), (2, 2)) }
		#[test] fn _13() { assert_eq!(snake_2d_split_u64(13), (3, 1)) }
		#[test] fn _14() { assert_eq!(snake_2d_split_u64(14), (4, 0)) }

		#[test] fn _15() { assert_eq!(snake_2d_split_u64(15), (0, 5)) }
		#[test] fn _16() { assert_eq!(snake_2d_split_u64(16), (1, 4)) }
		#[test] fn _17() { assert_eq!(snake_2d_split_u64(17), (2, 3)) }
		#[test] fn _18() { assert_eq!(snake_2d_split_u64(18), (3, 2)) }
		#[test] fn _19() { assert_eq!(snake_2d_split_u64(19), (4, 1)) }
		#[test] fn _20() { assert_eq!(snake_2d_split_u64(20), (5, 0)) }
	}

	mod unsplit_u64 {
		use super::*;
		#[test] fn _00() { assert_eq!(snake_2d_unsplit_u64(0, 0), 00) }

		#[test] fn _01() { assert_eq!(snake_2d_unsplit_u64(0, 1), 01) }
		#[test] fn _02() { assert_eq!(snake_2d_unsplit_u64(1, 0), 02) }

		#[test] fn _03() { assert_eq!(snake_2d_unsplit_u64(0, 2), 03) }
		#[test] fn _04() { assert_eq!(snake_2d_unsplit_u64(1, 1), 04) }
		#[test] fn _05() { assert_eq!(snake_2d_unsplit_u64(2, 0), 05) }

		#[test] fn _06() { assert_eq!(snake_2d_unsplit_u64(0, 3), 06) }
		#[test] fn _07() { assert_eq!(snake_2d_unsplit_u64(1, 2), 07) }
		#[test] fn _08() { assert_eq!(snake_2d_unsplit_u64(2, 1), 08) }
		#[test] fn _09() { assert_eq!(snake_2d_unsplit_u64(3, 0), 09) }

		#[test] fn _10() { assert_eq!(snake_2d_unsplit_u64(0, 4), 10) }
		#[test] fn _11() { assert_eq!(snake_2d_unsplit_u64(1, 3), 11) }
		#[test] fn _12() { assert_eq!(snake_2d_unsplit_u64(2, 2), 12) }
		#[test] fn _13() { assert_eq!(snake_2d_unsplit_u64(3, 1), 13) }
		#[test] fn _14() { assert_eq!(snake_2d_unsplit_u64(4, 0), 14) }

		#[test] fn _15() { assert_eq!(snake_2d_unsplit_u64(0, 5), 15) }
		#[test] fn _16() { assert_eq!(snake_2d_unsplit_u64(1, 4), 16) }
		#[test] fn _17() { assert_eq!(snake_2d_unsplit_u64(2, 3), 17) }
		#[test] fn _18() { assert_eq!(snake_2d_unsplit_u64(3, 2), 18) }
		#[test] fn _19() { assert_eq!(snake_2d_unsplit_u64(4, 1), 19) }
		#[test] fn _20() { assert_eq!(snake_2d_unsplit_u64(5, 0), 20) }
	}
}



#[cfg(test)]
mod expr {
	use super::*;

	mod from_int {
		use super::*;
		use Expr::*;
		#[test] fn _000() { assert_eq!(Expr::from_u64(000), Z) }
		#[test] fn _001() { assert_eq!(Expr::from_u64(001), PrevZ) }
		#[test] fn _002() { assert_eq!(Expr::from_u64(002), InitZ) }
		#[test] fn _003() { assert_eq!(Expr::from_u64(003), I) }

		#[test] fn _004() { assert_eq!(Expr::from_u64(004), UInt(0)) }
		#[test] fn _005() { assert_eq!(Expr::from_u64(005), Float(0.)) }
		#[test] fn _006() { assert_eq!(Expr::from_u64(006), Complex(cf(0., 0.))) }
		#[test] fn _007() { assert_eq!(Expr::from_u64(007), Neg(bx(Z))) }
		#[test] fn _008() { assert_eq!(Expr::from_u64(008), Abs(bx(Z))) }
		#[test] fn _009() { assert_eq!(Expr::from_u64(009), Arg(bx(Z))) }
		#[test] fn _010() { assert_eq!(Expr::from_u64(010), Re(bx(Z))) }
		#[test] fn _011() { assert_eq!(Expr::from_u64(011), Im(bx(Z))) }
		#[test] fn _012() { assert_eq!(Expr::from_u64(012), Conj(bx(Z))) }
		#[test] fn _013() { assert_eq!(Expr::from_u64(013), Exp(bx(Z))) }
		#[test] fn _014() { assert_eq!(Expr::from_u64(014), Ln(bx(Z))) }
		#[test] fn _015() { assert_eq!(Expr::from_u64(015), Sqrt(bx(Z))) }
		#[test] fn _016() { assert_eq!(Expr::from_u64(016), Sin(bx(Z))) }
		#[test] fn _017() { assert_eq!(Expr::from_u64(017), Cos(bx(Z))) }
		#[test] fn _018() { assert_eq!(Expr::from_u64(018), Tan(bx(Z))) }
		#[test] fn _019() { assert_eq!(Expr::from_u64(019), Sinh(bx(Z))) }
		#[test] fn _020() { assert_eq!(Expr::from_u64(020), Cosh(bx(Z))) }
		#[test] fn _021() { assert_eq!(Expr::from_u64(021), Tanh(bx(Z))) }
		#[test] fn _022() { assert_eq!(Expr::from_u64(022), Asin(bx(Z))) }
		#[test] fn _023() { assert_eq!(Expr::from_u64(023), Acos(bx(Z))) }
		#[test] fn _024() { assert_eq!(Expr::from_u64(024), Atan(bx(Z))) }
		#[test] fn _025() { assert_eq!(Expr::from_u64(025), Asinh(bx(Z))) }
		#[test] fn _026() { assert_eq!(Expr::from_u64(026), Acosh(bx(Z))) }
		#[test] fn _027() { assert_eq!(Expr::from_u64(027), Atanh(bx(Z))) }
		#[test] fn _028() { assert_eq!(Expr::from_u64(028), Round(bx(Z))) }
		#[test] fn _029() { assert_eq!(Expr::from_u64(029), Ceil(bx(Z))) }
		#[test] fn _030() { assert_eq!(Expr::from_u64(030), Floor(bx(Z))) }
		#[test] fn _031() { assert_eq!(Expr::from_u64(031), Sum(bx((Z, Z)))) }
		#[test] fn _032() { assert_eq!(Expr::from_u64(032), Prod(bx((Z, Z)))) }
		#[test] fn _033() { assert_eq!(Expr::from_u64(033), Div(bx((Z, Z)))) }
		#[test] fn _034() { assert_eq!(Expr::from_u64(034), Pow(bx((Z, Z)))) }

		#[test] fn _035() { assert_eq!(Expr::from_u64(035), UInt(1)) }
		#[test] fn _036() { assert_eq!(Expr::from_u64(036), Float(5e-324)) }
		#[test] fn _037() { assert_eq!(Expr::from_u64(037), Complex(cf(0., 1.401298464324817e-45))) }
		#[test] fn _038() { assert_eq!(Expr::from_u64(038), Neg(bx(PrevZ))) }
		#[test] fn _039() { assert_eq!(Expr::from_u64(039), Abs(bx(PrevZ))) }
		#[test] fn _040() { assert_eq!(Expr::from_u64(040), Arg(bx(PrevZ))) }
		#[test] fn _041() { assert_eq!(Expr::from_u64(041), Re(bx(PrevZ))) }
		#[test] fn _042() { assert_eq!(Expr::from_u64(042), Im(bx(PrevZ))) }
		#[test] fn _043() { assert_eq!(Expr::from_u64(043), Conj(bx(PrevZ))) }
		#[test] fn _044() { assert_eq!(Expr::from_u64(044), Exp(bx(PrevZ))) }
		#[test] fn _045() { assert_eq!(Expr::from_u64(045), Ln(bx(PrevZ))) }
		#[test] fn _046() { assert_eq!(Expr::from_u64(046), Sqrt(bx(PrevZ))) }
		#[test] fn _047() { assert_eq!(Expr::from_u64(047), Sin(bx(PrevZ))) }
		#[test] fn _048() { assert_eq!(Expr::from_u64(048), Cos(bx(PrevZ))) }
		#[test] fn _049() { assert_eq!(Expr::from_u64(049), Tan(bx(PrevZ))) }
		#[test] fn _050() { assert_eq!(Expr::from_u64(050), Sinh(bx(PrevZ))) }
		#[test] fn _051() { assert_eq!(Expr::from_u64(051), Cosh(bx(PrevZ))) }
		#[test] fn _052() { assert_eq!(Expr::from_u64(052), Tanh(bx(PrevZ))) }
		#[test] fn _053() { assert_eq!(Expr::from_u64(053), Asin(bx(PrevZ))) }
		#[test] fn _054() { assert_eq!(Expr::from_u64(054), Acos(bx(PrevZ))) }
		#[test] fn _055() { assert_eq!(Expr::from_u64(055), Atan(bx(PrevZ))) }
		#[test] fn _056() { assert_eq!(Expr::from_u64(056), Asinh(bx(PrevZ))) }
		#[test] fn _057() { assert_eq!(Expr::from_u64(057), Acosh(bx(PrevZ))) }
		#[test] fn _058() { assert_eq!(Expr::from_u64(058), Atanh(bx(PrevZ))) }
		#[test] fn _059() { assert_eq!(Expr::from_u64(059), Round(bx(PrevZ))) }
		#[test] fn _060() { assert_eq!(Expr::from_u64(060), Ceil(bx(PrevZ))) }
		#[test] fn _061() { assert_eq!(Expr::from_u64(061), Floor(bx(PrevZ))) }
		#[test] fn _062() { assert_eq!(Expr::from_u64(062), Sum(bx((Z, PrevZ)))) }
		#[test] fn _063() { assert_eq!(Expr::from_u64(063), Prod(bx((Z, PrevZ)))) }
		#[test] fn _064() { assert_eq!(Expr::from_u64(064), Div(bx((Z, PrevZ)))) }
		#[test] fn _065() { assert_eq!(Expr::from_u64(065), Pow(bx((Z, PrevZ)))) }

		#[test] fn _066() { assert_eq!(Expr::from_u64(066), UInt(2)) }
		#[test] fn _067() { assert_eq!(Expr::from_u64(067), Float(1e-323)) }
		#[test] fn _068() { assert_eq!(Expr::from_u64(068), Complex(cf(1.401298464324817e-45, 0.))) }
		#[test] fn _069() { assert_eq!(Expr::from_u64(069), Neg(bx(InitZ))) }
		#[test] fn _070() { assert_eq!(Expr::from_u64(070), Abs(bx(InitZ))) }
		#[test] fn _071() { assert_eq!(Expr::from_u64(071), Arg(bx(InitZ))) }
		#[test] fn _072() { assert_eq!(Expr::from_u64(072), Re(bx(InitZ))) }
		#[test] fn _073() { assert_eq!(Expr::from_u64(073), Im(bx(InitZ))) }
		#[test] fn _074() { assert_eq!(Expr::from_u64(074), Conj(bx(InitZ))) }
		#[test] fn _075() { assert_eq!(Expr::from_u64(075), Exp(bx(InitZ))) }
		#[test] fn _076() { assert_eq!(Expr::from_u64(076), Ln(bx(InitZ))) }
		#[test] fn _077() { assert_eq!(Expr::from_u64(077), Sqrt(bx(InitZ))) }
		#[test] fn _078() { assert_eq!(Expr::from_u64(078), Sin(bx(InitZ))) }
		#[test] fn _079() { assert_eq!(Expr::from_u64(079), Cos(bx(InitZ))) }
		#[test] fn _080() { assert_eq!(Expr::from_u64(080), Tan(bx(InitZ))) }
		#[test] fn _081() { assert_eq!(Expr::from_u64(081), Sinh(bx(InitZ))) }
		#[test] fn _082() { assert_eq!(Expr::from_u64(082), Cosh(bx(InitZ))) }
		#[test] fn _083() { assert_eq!(Expr::from_u64(083), Tanh(bx(InitZ))) }
		#[test] fn _084() { assert_eq!(Expr::from_u64(084), Asin(bx(InitZ))) }
		#[test] fn _085() { assert_eq!(Expr::from_u64(085), Acos(bx(InitZ))) }
		#[test] fn _086() { assert_eq!(Expr::from_u64(086), Atan(bx(InitZ))) }
		#[test] fn _087() { assert_eq!(Expr::from_u64(087), Asinh(bx(InitZ))) }
		#[test] fn _088() { assert_eq!(Expr::from_u64(088), Acosh(bx(InitZ))) }
		#[test] fn _089() { assert_eq!(Expr::from_u64(089), Atanh(bx(InitZ))) }
		#[test] fn _090() { assert_eq!(Expr::from_u64(090), Round(bx(InitZ))) }
		#[test] fn _091() { assert_eq!(Expr::from_u64(091), Ceil(bx(InitZ))) }
		#[test] fn _092() { assert_eq!(Expr::from_u64(092), Floor(bx(InitZ))) }
		#[test] fn _093() { assert_eq!(Expr::from_u64(093), Sum(bx((PrevZ, Z)))) }
		#[test] fn _094() { assert_eq!(Expr::from_u64(094), Prod(bx((PrevZ, Z)))) }
		#[test] fn _095() { assert_eq!(Expr::from_u64(095), Div(bx((PrevZ, Z)))) }
		#[test] fn _096() { assert_eq!(Expr::from_u64(096), Pow(bx((PrevZ, Z)))) }

		#[test] fn _097() { assert_eq!(Expr::from_u64(097), UInt(3)) }
		#[test] fn _098() { assert_eq!(Expr::from_u64(098), Float(1.5e-323)) }
		#[test] fn _099() { assert_eq!(Expr::from_u64(099), Complex(cf(0., 2.802596928649634e-45))) }
		#[test] fn _100() { assert_eq!(Expr::from_u64(100), Neg(bx(I))) }
		#[test] fn _101() { assert_eq!(Expr::from_u64(101), Abs(bx(I))) }
		#[test] fn _102() { assert_eq!(Expr::from_u64(102), Arg(bx(I))) }
		#[test] fn _103() { assert_eq!(Expr::from_u64(103), Re(bx(I))) }
		#[test] fn _104() { assert_eq!(Expr::from_u64(104), Im(bx(I))) }
		#[test] fn _105() { assert_eq!(Expr::from_u64(105), Conj(bx(I))) }
		#[test] fn _106() { assert_eq!(Expr::from_u64(106), Exp(bx(I))) }
		#[test] fn _107() { assert_eq!(Expr::from_u64(107), Ln(bx(I))) }
		#[test] fn _108() { assert_eq!(Expr::from_u64(108), Sqrt(bx(I))) }
		#[test] fn _109() { assert_eq!(Expr::from_u64(109), Sin(bx(I))) }
		#[test] fn _110() { assert_eq!(Expr::from_u64(110), Cos(bx(I))) }
		#[test] fn _111() { assert_eq!(Expr::from_u64(111), Tan(bx(I))) }
		#[test] fn _112() { assert_eq!(Expr::from_u64(112), Sinh(bx(I))) }
		#[test] fn _113() { assert_eq!(Expr::from_u64(113), Cosh(bx(I))) }
		#[test] fn _114() { assert_eq!(Expr::from_u64(114), Tanh(bx(I))) }
		#[test] fn _115() { assert_eq!(Expr::from_u64(115), Asin(bx(I))) }
		#[test] fn _116() { assert_eq!(Expr::from_u64(116), Acos(bx(I))) }
		#[test] fn _117() { assert_eq!(Expr::from_u64(117), Atan(bx(I))) }
		#[test] fn _118() { assert_eq!(Expr::from_u64(118), Asinh(bx(I))) }
		#[test] fn _119() { assert_eq!(Expr::from_u64(119), Acosh(bx(I))) }
		#[test] fn _120() { assert_eq!(Expr::from_u64(120), Atanh(bx(I))) }
		#[test] fn _121() { assert_eq!(Expr::from_u64(121), Round(bx(I))) }
		#[test] fn _122() { assert_eq!(Expr::from_u64(122), Ceil(bx(I))) }
		#[test] fn _123() { assert_eq!(Expr::from_u64(123), Floor(bx(I))) }
		#[test] fn _124() { assert_eq!(Expr::from_u64(124), Sum(bx((Z, InitZ)))) }
		#[test] fn _125() { assert_eq!(Expr::from_u64(125), Prod(bx((Z, InitZ)))) }
		#[test] fn _126() { assert_eq!(Expr::from_u64(126), Div(bx((Z, InitZ)))) }
		#[test] fn _127() { assert_eq!(Expr::from_u64(127), Pow(bx((Z, InitZ)))) }

		#[test] fn _128() { assert_eq!(Expr::from_u64(128), UInt(4)) }
		#[test] fn _129() { assert_eq!(Expr::from_u64(129), Float(2e-323)) }
		#[test] fn _130() { assert_eq!(Expr::from_u64(130), Complex(cf(1.401298464324817e-45, 1.401298464324817e-45))) }
		#[test] fn _131() { assert_eq!(Expr::from_u64(131), Neg(bx(UInt(0)))) }
		#[test] fn _132() { assert_eq!(Expr::from_u64(132), Abs(bx(UInt(0)))) }
		#[test] fn _133() { assert_eq!(Expr::from_u64(133), Arg(bx(UInt(0)))) }
		#[test] fn _134() { assert_eq!(Expr::from_u64(134), Re(bx(UInt(0)))) }
		#[test] fn _135() { assert_eq!(Expr::from_u64(135), Im(bx(UInt(0)))) }
		#[test] fn _136() { assert_eq!(Expr::from_u64(136), Conj(bx(UInt(0)))) }
		#[test] fn _137() { assert_eq!(Expr::from_u64(137), Exp(bx(UInt(0)))) }
		#[test] fn _138() { assert_eq!(Expr::from_u64(138), Ln(bx(UInt(0)))) }
		#[test] fn _139() { assert_eq!(Expr::from_u64(139), Sqrt(bx(UInt(0)))) }
		#[test] fn _140() { assert_eq!(Expr::from_u64(140), Sin(bx(UInt(0)))) }
		#[test] fn _141() { assert_eq!(Expr::from_u64(141), Cos(bx(UInt(0)))) }
		#[test] fn _142() { assert_eq!(Expr::from_u64(142), Tan(bx(UInt(0)))) }
		#[test] fn _143() { assert_eq!(Expr::from_u64(143), Sinh(bx(UInt(0)))) }
		#[test] fn _144() { assert_eq!(Expr::from_u64(144), Cosh(bx(UInt(0)))) }
		#[test] fn _145() { assert_eq!(Expr::from_u64(145), Tanh(bx(UInt(0)))) }
		#[test] fn _146() { assert_eq!(Expr::from_u64(146), Asin(bx(UInt(0)))) }
		#[test] fn _147() { assert_eq!(Expr::from_u64(147), Acos(bx(UInt(0)))) }
		#[test] fn _148() { assert_eq!(Expr::from_u64(148), Atan(bx(UInt(0)))) }
		#[test] fn _149() { assert_eq!(Expr::from_u64(149), Asinh(bx(UInt(0)))) }
		#[test] fn _150() { assert_eq!(Expr::from_u64(150), Acosh(bx(UInt(0)))) }
		#[test] fn _151() { assert_eq!(Expr::from_u64(151), Atanh(bx(UInt(0)))) }
		#[test] fn _152() { assert_eq!(Expr::from_u64(152), Round(bx(UInt(0)))) }
		#[test] fn _153() { assert_eq!(Expr::from_u64(153), Ceil(bx(UInt(0)))) }
		#[test] fn _154() { assert_eq!(Expr::from_u64(154), Floor(bx(UInt(0)))) }
		#[test] fn _155() { assert_eq!(Expr::from_u64(155), Sum(bx((PrevZ, PrevZ)))) }
		#[test] fn _156() { assert_eq!(Expr::from_u64(156), Prod(bx((PrevZ, PrevZ)))) }
		#[test] fn _157() { assert_eq!(Expr::from_u64(157), Div(bx((PrevZ, PrevZ)))) }
		#[test] fn _158() { assert_eq!(Expr::from_u64(158), Pow(bx((PrevZ, PrevZ)))) }

		#[test] fn _19468() { assert_eq!(Expr::from_u64(19468), Sum(bx((Prod(bx((Z, Z))), InitZ)))) }
	}

	mod to_int {
		use super::*;
		use Expr::*;
		#[test] fn _000() { assert_eq!(BigUint::from(000_u32), Z.to_int()) }
		#[test] fn _001() { assert_eq!(BigUint::from(001_u32), PrevZ.to_int()) }
		#[test] fn _002() { assert_eq!(BigUint::from(002_u32), InitZ.to_int()) }
		#[test] fn _003() { assert_eq!(BigUint::from(003_u32), I.to_int()) }

		#[test] fn _004() { assert_eq!(BigUint::from(004_u32), UInt(0).to_int()) }
		#[test] fn _005() { assert_eq!(BigUint::from(005_u32), Float(0.).to_int()) }
		#[test] fn _006() { assert_eq!(BigUint::from(006_u32), Complex(cf(0., 0.)).to_int()) }
		#[test] fn _007() { assert_eq!(BigUint::from(007_u32), Neg(bx(Z)).to_int()) }
		#[test] fn _008() { assert_eq!(BigUint::from(008_u32), Abs(bx(Z)).to_int()) }
		#[test] fn _009() { assert_eq!(BigUint::from(009_u32), Arg(bx(Z)).to_int()) }
		#[test] fn _010() { assert_eq!(BigUint::from(010_u32), Re(bx(Z)).to_int()) }
		#[test] fn _011() { assert_eq!(BigUint::from(011_u32), Im(bx(Z)).to_int()) }
		#[test] fn _012() { assert_eq!(BigUint::from(012_u32), Conj(bx(Z)).to_int()) }
		#[test] fn _013() { assert_eq!(BigUint::from(013_u32), Exp(bx(Z)).to_int()) }
		#[test] fn _014() { assert_eq!(BigUint::from(014_u32), Ln(bx(Z)).to_int()) }
		#[test] fn _015() { assert_eq!(BigUint::from(015_u32), Sqrt(bx(Z)).to_int()) }
		#[test] fn _016() { assert_eq!(BigUint::from(016_u32), Sin(bx(Z)).to_int()) }
		#[test] fn _017() { assert_eq!(BigUint::from(017_u32), Cos(bx(Z)).to_int()) }
		#[test] fn _018() { assert_eq!(BigUint::from(018_u32), Tan(bx(Z)).to_int()) }
		#[test] fn _019() { assert_eq!(BigUint::from(019_u32), Sinh(bx(Z)).to_int()) }
		#[test] fn _020() { assert_eq!(BigUint::from(020_u32), Cosh(bx(Z)).to_int()) }
		#[test] fn _021() { assert_eq!(BigUint::from(021_u32), Tanh(bx(Z)).to_int()) }
		#[test] fn _022() { assert_eq!(BigUint::from(022_u32), Asin(bx(Z)).to_int()) }
		#[test] fn _023() { assert_eq!(BigUint::from(023_u32), Acos(bx(Z)).to_int()) }
		#[test] fn _024() { assert_eq!(BigUint::from(024_u32), Atan(bx(Z)).to_int()) }
		#[test] fn _025() { assert_eq!(BigUint::from(025_u32), Asinh(bx(Z)).to_int()) }
		#[test] fn _026() { assert_eq!(BigUint::from(026_u32), Acosh(bx(Z)).to_int()) }
		#[test] fn _027() { assert_eq!(BigUint::from(027_u32), Atanh(bx(Z)).to_int()) }
		#[test] fn _028() { assert_eq!(BigUint::from(028_u32), Round(bx(Z)).to_int()) }
		#[test] fn _029() { assert_eq!(BigUint::from(029_u32), Ceil(bx(Z)).to_int()) }
		#[test] fn _030() { assert_eq!(BigUint::from(030_u32), Floor(bx(Z)).to_int()) }
		#[test] fn _031() { assert_eq!(BigUint::from(031_u32), Sum(bx((Z, Z))).to_int()) }
		#[test] fn _032() { assert_eq!(BigUint::from(032_u32), Prod(bx((Z, Z))).to_int()) }
		#[test] fn _033() { assert_eq!(BigUint::from(033_u32), Div(bx((Z, Z))).to_int()) }
		#[test] fn _034() { assert_eq!(BigUint::from(034_u32), Pow(bx((Z, Z))).to_int()) }

		#[test] fn _035() { assert_eq!(BigUint::from(035_u32), UInt(1).to_int()) }
		#[test] fn _036() { assert_eq!(BigUint::from(036_u32), Float(5e-324).to_int()) }
		#[test] fn _037() { assert_eq!(BigUint::from(037_u32), Complex(cf(0., 1.401298464324817e-45)).to_int()) }
		#[test] fn _038() { assert_eq!(BigUint::from(038_u32), Neg(bx(PrevZ)).to_int()) }
		#[test] fn _039() { assert_eq!(BigUint::from(039_u32), Abs(bx(PrevZ)).to_int()) }
		#[test] fn _040() { assert_eq!(BigUint::from(040_u32), Arg(bx(PrevZ)).to_int()) }
		#[test] fn _041() { assert_eq!(BigUint::from(041_u32), Re(bx(PrevZ)).to_int()) }
		#[test] fn _042() { assert_eq!(BigUint::from(042_u32), Im(bx(PrevZ)).to_int()) }
		#[test] fn _043() { assert_eq!(BigUint::from(043_u32), Conj(bx(PrevZ)).to_int()) }
		#[test] fn _044() { assert_eq!(BigUint::from(044_u32), Exp(bx(PrevZ)).to_int()) }
		#[test] fn _045() { assert_eq!(BigUint::from(045_u32), Ln(bx(PrevZ)).to_int()) }
		#[test] fn _046() { assert_eq!(BigUint::from(046_u32), Sqrt(bx(PrevZ)).to_int()) }
		#[test] fn _047() { assert_eq!(BigUint::from(047_u32), Sin(bx(PrevZ)).to_int()) }
		#[test] fn _048() { assert_eq!(BigUint::from(048_u32), Cos(bx(PrevZ)).to_int()) }
		#[test] fn _049() { assert_eq!(BigUint::from(049_u32), Tan(bx(PrevZ)).to_int()) }
		#[test] fn _050() { assert_eq!(BigUint::from(050_u32), Sinh(bx(PrevZ)).to_int()) }
		#[test] fn _051() { assert_eq!(BigUint::from(051_u32), Cosh(bx(PrevZ)).to_int()) }
		#[test] fn _052() { assert_eq!(BigUint::from(052_u32), Tanh(bx(PrevZ)).to_int()) }
		#[test] fn _053() { assert_eq!(BigUint::from(053_u32), Asin(bx(PrevZ)).to_int()) }
		#[test] fn _054() { assert_eq!(BigUint::from(054_u32), Acos(bx(PrevZ)).to_int()) }
		#[test] fn _055() { assert_eq!(BigUint::from(055_u32), Atan(bx(PrevZ)).to_int()) }
		#[test] fn _056() { assert_eq!(BigUint::from(056_u32), Asinh(bx(PrevZ)).to_int()) }
		#[test] fn _057() { assert_eq!(BigUint::from(057_u32), Acosh(bx(PrevZ)).to_int()) }
		#[test] fn _058() { assert_eq!(BigUint::from(058_u32), Atanh(bx(PrevZ)).to_int()) }
		#[test] fn _059() { assert_eq!(BigUint::from(059_u32), Round(bx(PrevZ)).to_int()) }
		#[test] fn _060() { assert_eq!(BigUint::from(060_u32), Ceil(bx(PrevZ)).to_int()) }
		#[test] fn _061() { assert_eq!(BigUint::from(061_u32), Floor(bx(PrevZ)).to_int()) }
		#[test] fn _062() { assert_eq!(BigUint::from(062_u32), Sum(bx((Z, PrevZ))).to_int()) }
		#[test] fn _063() { assert_eq!(BigUint::from(063_u32), Prod(bx((Z, PrevZ))).to_int()) }
		#[test] fn _064() { assert_eq!(BigUint::from(064_u32), Div(bx((Z, PrevZ))).to_int()) }
		#[test] fn _065() { assert_eq!(BigUint::from(065_u32), Pow(bx((Z, PrevZ))).to_int()) }

		#[test] fn _066() { assert_eq!(BigUint::from(066_u32), UInt(2).to_int()) }
		#[test] fn _067() { assert_eq!(BigUint::from(067_u32), Float(1e-323).to_int()) }
		#[test] fn _068() { assert_eq!(BigUint::from(068_u32), Complex(cf(1.401298464324817e-45, 0.)).to_int()) }
		#[test] fn _069() { assert_eq!(BigUint::from(069_u32), Neg(bx(InitZ)).to_int()) }
		#[test] fn _070() { assert_eq!(BigUint::from(070_u32), Abs(bx(InitZ)).to_int()) }
		#[test] fn _071() { assert_eq!(BigUint::from(071_u32), Arg(bx(InitZ)).to_int()) }
		#[test] fn _072() { assert_eq!(BigUint::from(072_u32), Re(bx(InitZ)).to_int()) }
		#[test] fn _073() { assert_eq!(BigUint::from(073_u32), Im(bx(InitZ)).to_int()) }
		#[test] fn _074() { assert_eq!(BigUint::from(074_u32), Conj(bx(InitZ)).to_int()) }
		#[test] fn _075() { assert_eq!(BigUint::from(075_u32), Exp(bx(InitZ)).to_int()) }
		#[test] fn _076() { assert_eq!(BigUint::from(076_u32), Ln(bx(InitZ)).to_int()) }
		#[test] fn _077() { assert_eq!(BigUint::from(077_u32), Sqrt(bx(InitZ)).to_int()) }
		#[test] fn _078() { assert_eq!(BigUint::from(078_u32), Sin(bx(InitZ)).to_int()) }
		#[test] fn _079() { assert_eq!(BigUint::from(079_u32), Cos(bx(InitZ)).to_int()) }
		#[test] fn _080() { assert_eq!(BigUint::from(080_u32), Tan(bx(InitZ)).to_int()) }
		#[test] fn _081() { assert_eq!(BigUint::from(081_u32), Sinh(bx(InitZ)).to_int()) }
		#[test] fn _082() { assert_eq!(BigUint::from(082_u32), Cosh(bx(InitZ)).to_int()) }
		#[test] fn _083() { assert_eq!(BigUint::from(083_u32), Tanh(bx(InitZ)).to_int()) }
		#[test] fn _084() { assert_eq!(BigUint::from(084_u32), Asin(bx(InitZ)).to_int()) }
		#[test] fn _085() { assert_eq!(BigUint::from(085_u32), Acos(bx(InitZ)).to_int()) }
		#[test] fn _086() { assert_eq!(BigUint::from(086_u32), Atan(bx(InitZ)).to_int()) }
		#[test] fn _087() { assert_eq!(BigUint::from(087_u32), Asinh(bx(InitZ)).to_int()) }
		#[test] fn _088() { assert_eq!(BigUint::from(088_u32), Acosh(bx(InitZ)).to_int()) }
		#[test] fn _089() { assert_eq!(BigUint::from(089_u32), Atanh(bx(InitZ)).to_int()) }
		#[test] fn _090() { assert_eq!(BigUint::from(090_u32), Round(bx(InitZ)).to_int()) }
		#[test] fn _091() { assert_eq!(BigUint::from(091_u32), Ceil(bx(InitZ)).to_int()) }
		#[test] fn _092() { assert_eq!(BigUint::from(092_u32), Floor(bx(InitZ)).to_int()) }
		#[test] fn _093() { assert_eq!(BigUint::from(093_u32), Sum(bx((PrevZ, Z))).to_int()) }
		#[test] fn _094() { assert_eq!(BigUint::from(094_u32), Prod(bx((PrevZ, Z))).to_int()) }
		#[test] fn _095() { assert_eq!(BigUint::from(095_u32), Div(bx((PrevZ, Z))).to_int()) }
		#[test] fn _096() { assert_eq!(BigUint::from(096_u32), Pow(bx((PrevZ, Z))).to_int()) }

		#[test] fn _097() { assert_eq!(BigUint::from(097_u32), UInt(3).to_int()) }
		#[test] fn _098() { assert_eq!(BigUint::from(098_u32), Float(1.5e-323).to_int()) }
		#[test] fn _099() { assert_eq!(BigUint::from(099_u32), Complex(cf(0., 2.802596928649634e-45)).to_int()) }
		#[test] fn _100() { assert_eq!(BigUint::from(100_u32), Neg(bx(I)).to_int()) }
		#[test] fn _101() { assert_eq!(BigUint::from(101_u32), Abs(bx(I)).to_int()) }
		#[test] fn _102() { assert_eq!(BigUint::from(102_u32), Arg(bx(I)).to_int()) }
		#[test] fn _103() { assert_eq!(BigUint::from(103_u32), Re(bx(I)).to_int()) }
		#[test] fn _104() { assert_eq!(BigUint::from(104_u32), Im(bx(I)).to_int()) }
		#[test] fn _105() { assert_eq!(BigUint::from(105_u32), Conj(bx(I)).to_int()) }
		#[test] fn _106() { assert_eq!(BigUint::from(106_u32), Exp(bx(I)).to_int()) }
		#[test] fn _107() { assert_eq!(BigUint::from(107_u32), Ln(bx(I)).to_int()) }
		#[test] fn _108() { assert_eq!(BigUint::from(108_u32), Sqrt(bx(I)).to_int()) }
		#[test] fn _109() { assert_eq!(BigUint::from(109_u32), Sin(bx(I)).to_int()) }
		#[test] fn _110() { assert_eq!(BigUint::from(110_u32), Cos(bx(I)).to_int()) }
		#[test] fn _111() { assert_eq!(BigUint::from(111_u32), Tan(bx(I)).to_int()) }
		#[test] fn _112() { assert_eq!(BigUint::from(112_u32), Sinh(bx(I)).to_int()) }
		#[test] fn _113() { assert_eq!(BigUint::from(113_u32), Cosh(bx(I)).to_int()) }
		#[test] fn _114() { assert_eq!(BigUint::from(114_u32), Tanh(bx(I)).to_int()) }
		#[test] fn _115() { assert_eq!(BigUint::from(115_u32), Asin(bx(I)).to_int()) }
		#[test] fn _116() { assert_eq!(BigUint::from(116_u32), Acos(bx(I)).to_int()) }
		#[test] fn _117() { assert_eq!(BigUint::from(117_u32), Atan(bx(I)).to_int()) }
		#[test] fn _118() { assert_eq!(BigUint::from(118_u32), Asinh(bx(I)).to_int()) }
		#[test] fn _119() { assert_eq!(BigUint::from(119_u32), Acosh(bx(I)).to_int()) }
		#[test] fn _120() { assert_eq!(BigUint::from(120_u32), Atanh(bx(I)).to_int()) }
		#[test] fn _121() { assert_eq!(BigUint::from(121_u32), Round(bx(I)).to_int()) }
		#[test] fn _122() { assert_eq!(BigUint::from(122_u32), Ceil(bx(I)).to_int()) }
		#[test] fn _123() { assert_eq!(BigUint::from(123_u32), Floor(bx(I)).to_int()) }
		#[test] fn _124() { assert_eq!(BigUint::from(124_u32), Sum(bx((Z, InitZ))).to_int()) }
		#[test] fn _125() { assert_eq!(BigUint::from(125_u32), Prod(bx((Z, InitZ))).to_int()) }
		#[test] fn _126() { assert_eq!(BigUint::from(126_u32), Div(bx((Z, InitZ))).to_int()) }
		#[test] fn _127() { assert_eq!(BigUint::from(127_u32), Pow(bx((Z, InitZ))).to_int()) }

		#[test] fn _128() { assert_eq!(BigUint::from(128_u32), UInt(4).to_int()) }
		#[test] fn _129() { assert_eq!(BigUint::from(129_u32), Float(2e-323).to_int()) }
		#[test] fn _130() { assert_eq!(BigUint::from(130_u32), Complex(cf(1.401298464324817e-45, 1.401298464324817e-45)).to_int()) }
		#[test] fn _131() { assert_eq!(BigUint::from(131_u32), Neg(bx(UInt(0))).to_int()) }
		#[test] fn _132() { assert_eq!(BigUint::from(132_u32), Abs(bx(UInt(0))).to_int()) }
		#[test] fn _133() { assert_eq!(BigUint::from(133_u32), Arg(bx(UInt(0))).to_int()) }
		#[test] fn _134() { assert_eq!(BigUint::from(134_u32), Re(bx(UInt(0))).to_int()) }
		#[test] fn _135() { assert_eq!(BigUint::from(135_u32), Im(bx(UInt(0))).to_int()) }
		#[test] fn _136() { assert_eq!(BigUint::from(136_u32), Conj(bx(UInt(0))).to_int()) }
		#[test] fn _137() { assert_eq!(BigUint::from(137_u32), Exp(bx(UInt(0))).to_int()) }
		#[test] fn _138() { assert_eq!(BigUint::from(138_u32), Ln(bx(UInt(0))).to_int()) }
		#[test] fn _139() { assert_eq!(BigUint::from(139_u32), Sqrt(bx(UInt(0))).to_int()) }
		#[test] fn _140() { assert_eq!(BigUint::from(140_u32), Sin(bx(UInt(0))).to_int()) }
		#[test] fn _141() { assert_eq!(BigUint::from(141_u32), Cos(bx(UInt(0))).to_int()) }
		#[test] fn _142() { assert_eq!(BigUint::from(142_u32), Tan(bx(UInt(0))).to_int()) }
		#[test] fn _143() { assert_eq!(BigUint::from(143_u32), Sinh(bx(UInt(0))).to_int()) }
		#[test] fn _144() { assert_eq!(BigUint::from(144_u32), Cosh(bx(UInt(0))).to_int()) }
		#[test] fn _145() { assert_eq!(BigUint::from(145_u32), Tanh(bx(UInt(0))).to_int()) }
		#[test] fn _146() { assert_eq!(BigUint::from(146_u32), Asin(bx(UInt(0))).to_int()) }
		#[test] fn _147() { assert_eq!(BigUint::from(147_u32), Acos(bx(UInt(0))).to_int()) }
		#[test] fn _148() { assert_eq!(BigUint::from(148_u32), Atan(bx(UInt(0))).to_int()) }
		#[test] fn _149() { assert_eq!(BigUint::from(149_u32), Asinh(bx(UInt(0))).to_int()) }
		#[test] fn _150() { assert_eq!(BigUint::from(150_u32), Acosh(bx(UInt(0))).to_int()) }
		#[test] fn _151() { assert_eq!(BigUint::from(151_u32), Atanh(bx(UInt(0))).to_int()) }
		#[test] fn _152() { assert_eq!(BigUint::from(152_u32), Round(bx(UInt(0))).to_int()) }
		#[test] fn _153() { assert_eq!(BigUint::from(153_u32), Ceil(bx(UInt(0))).to_int()) }
		#[test] fn _154() { assert_eq!(BigUint::from(154_u32), Floor(bx(UInt(0))).to_int()) }
		#[test] fn _155() { assert_eq!(BigUint::from(155_u32), Sum(bx((PrevZ, PrevZ))).to_int()) }
		#[test] fn _156() { assert_eq!(BigUint::from(156_u32), Prod(bx((PrevZ, PrevZ))).to_int()) }
		#[test] fn _157() { assert_eq!(BigUint::from(157_u32), Div(bx((PrevZ, PrevZ))).to_int()) }
		#[test] fn _158() { assert_eq!(BigUint::from(158_u32), Pow(bx((PrevZ, PrevZ))).to_int()) }

		#[test] fn _19468() { assert_eq!(BigUint::from(19468_u32), Sum(bx((Prod(bx((Z, Z))), InitZ))).to_int()) }
	}

	mod from_str {
		use super::*;
		use Expr::*;

		#[test] fn z() { assert_eq!(Expr::from_str("z"), Ok(Z)) }
		#[test] fn int_42() { assert_eq!(Expr::from_str("42"), Ok(UInt(42))) }
		#[test] fn sin_sin_z() { assert_eq!(Expr::from_str("sin(sin(z))"), Ok(Sin(bx(Sin(bx(Z)))))) }
		#[test] fn sin_cos_z() { assert_eq!(Expr::from_str("sin(cos(z))"), Ok(Sin(bx(Cos(bx(Z)))))) }
		#[test] fn Sin_Cos_z() { assert_eq!(Expr::from_str("Sin(Cos(z))"), Ok(Sin(bx(Cos(bx(Z)))))) }
		#[test] fn sum_z_3 () { assert_eq!(Expr::from_str("z+3"), Ok(Sum(bx((Z, UInt(3)))))) }
		#[test] fn prod_z_3() { assert_eq!(Expr::from_str("z*3"), Ok(Prod(bx((Z, UInt(3)))))) }
		#[test] fn pow_z_3 () { assert_eq!(Expr::from_str("z^3"), Ok(Pow(bx((Z, UInt(3)))))) }
		#[test] fn z_sin2z_sq_plus_1() { assert_eq!(Expr::from_str("z*sin(2*z)^2 + 1"), Ok(Sum(bx((Prod(bx((Z, Pow(bx((Sin(bx(Prod(bx((UInt(2), Z))))), UInt(2))))))), UInt(1)))))) }

		#[test] fn neg_z() { assert_eq!(Expr::from_str("-z"), Ok(Neg(bx(Z)))) }
		#[test] fn abs_z() { assert_eq!(Expr::from_str("abs(z)"), Ok(Abs(bx(Z)))) }
		#[test] fn arg_z() { assert_eq!(Expr::from_str("arg(z)"), Ok(Arg(bx(Z)))) }
		#[test] fn re_z() { assert_eq!(Expr::from_str("re(z)"), Ok(Re(bx(Z)))) }
		#[test] fn im_z() { assert_eq!(Expr::from_str("im(z)"), Ok(Im(bx(Z)))) }
		#[test] fn conj_z() { assert_eq!(Expr::from_str("conj(z)"), Ok(Conj(bx(Z)))) }
		#[test] fn exp_z() { assert_eq!(Expr::from_str("exp(z)"), Ok(Exp(bx(Z)))) }
		#[test] fn ln_z() { assert_eq!(Expr::from_str("ln(z)"), Ok(Ln(bx(Z)))) }
		#[test] fn sqrt_z() { assert_eq!(Expr::from_str("sqrt(z)"), Ok(Sqrt(bx(Z)))) }

		#[test] fn sin_z() { assert_eq!(Expr::from_str("sin(z)"), Ok(Sin(bx(Z)))) }
		#[test] fn cos_z() { assert_eq!(Expr::from_str("cos(z)"), Ok(Cos(bx(Z)))) }
		#[test] fn tan_z() { assert_eq!(Expr::from_str("tan(z)"), Ok(Tan(bx(Z)))) }
		#[test] fn asin_z() { assert_eq!(Expr::from_str("asin(z)"), Ok(Asin(bx(Z)))) }
		#[test] fn acos_z() { assert_eq!(Expr::from_str("acos(z)"), Ok(Acos(bx(Z)))) }
		#[test] fn atan_z() { assert_eq!(Expr::from_str("atan(z)"), Ok(Atan(bx(Z)))) }

		#[test] fn sinh_z() { assert_eq!(Expr::from_str("sinh(z)"), Ok(Sinh(bx(Z)))) }
		#[test] fn cosh_z() { assert_eq!(Expr::from_str("cosh(z)"), Ok(Cosh(bx(Z)))) }
		#[test] fn tanh_z() { assert_eq!(Expr::from_str("tanh(z)"), Ok(Tanh(bx(Z)))) }
		#[test] fn asinh_z() { assert_eq!(Expr::from_str("asinh(z)"), Ok(Asinh(bx(Z)))) }
		#[test] fn acosh_z() { assert_eq!(Expr::from_str("acosh(z)"), Ok(Acosh(bx(Z)))) }
		#[test] fn atanh_z() { assert_eq!(Expr::from_str("atanh(z)"), Ok(Atanh(bx(Z)))) }

		#[test] fn round_z() { assert_eq!(Expr::from_str("round(z)"), Ok(Round(bx(Z)))) }
		#[test] fn ceil_z() { assert_eq!(Expr::from_str("ceil(z)"), Ok(Ceil(bx(Z)))) }
		#[test] fn floor_z() { assert_eq!(Expr::from_str("floor(z)"), Ok(Floor(bx(Z)))) }
	}
}
