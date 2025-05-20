//! Render random fractals >:3

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
use num::{complex::{Complex64, ComplexFloat}, BigUint, One, Zero};
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
		{about}\n\
		Author: {author}\n\
		\n\
		{usage-heading} {usage}\n\
		\n\
		{all-args}{after-help}\
	",
)]
struct CliArgs {
	fractal_id: Option<String>,

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
	// #[arg(short='q', long, default_value_t=false)]
	// high_quality: bool,

	// TODO
	#[arg(short='r', long, default_value_t=false)]
	keys_repeat: bool,
}

struct Params {
	fractal: (BigUint, Expr),
	assign_zesc_once: bool,
	break_loop: bool,
	zesc_value: float,
	quality: Quality,
	keys_repeat: bool,
}
impl From<CliArgs> for Params {
	fn from(CliArgs {
		fractal_id,
		assign_zesc_once,
		break_loop,
		zesc_value,
		// high_quality,
		keys_repeat,
	}: CliArgs) -> Self {
		Self {
			fractal: {
				if let Some(fractal_id) = fractal_id {
					let id = BigUint::from_str(&fractal_id).unwrap();
					let expr = Expr::from_int(id.clone());
					(id, expr)
				} else {
					let mut rng = rng();
					loop {
						const N_MAX_DIGITS: u32 = 19;
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
			quality: Quality(0),
			keys_repeat,
		}
	}
}



fn main() {
	// let fractal_4768_mandelbrot: Expr = {
	// 	use Expr::*;
	// 	Sum(bx((Prod(bx((Z, Z))), InitZ)))
	// };
	// let fractal_898512: Expr = {
	// 	use Expr::*;
	// 	// (sin(Z))^(sin(cosh(Z)))
	// 	Pow(bx((Sin(bx(Z)), Sin(bx(Cosh(bx(Z)))))))
	// };
	// let fractal_x: Expr = {
	// 	use Expr::*;
	// 	// 7134919 -> cosh(0e0+0e0i)/cosh(Z/PrevZ)
	// 	// Div(bx((Cosh(bx(UInt(0_u32.into()))), Cosh(bx(Div(bx((Z, PrevZ)))))))) // 6522302
	// 	Div(bx((UInt(1_u32.into()), Cosh(bx(Div(bx((Z, PrevZ)))))))) // 5758027
	// };
	// for n in 0_u128.. {
	// 	// let n: BigUint = prompt("Expr N: ").parse().unwrap();
	// 	let id = BigUint::from(n);
	// 	let expr = Expr::from_int(id.clone());
	// 	println!("{} -> {}", id, expr.to_string());
	// 	if expr == fractal_x { break }
	// }
	// #[allow(unreachable_code)]
	// return;

	let cli_args = CliArgs::parse();

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
		},
	).unwrap();

	window.set_target_fps(60);
	window.update_with_buffer(&buffer, w, h).unwrap();

	let mut zoom : float = 1.0;
	let mut cam_x: float = 0.0;
	let mut cam_y: float = 0.0;

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
		}

		if window.is_key_pressed_or_down(Key::Q, params.keys_repeat) {
			params.quality.decrease();
			println!("quality: {:?}", params.quality);
			is_redraw_needed = true;
		}
		if window.is_key_pressed_or_down(Key::E, params.keys_repeat) {
			params.quality.increase();
			println!("quality: {:?}", params.quality);
			is_redraw_needed = true;
		}

		if window.is_key_pressed_or_down(Key::N, params.keys_repeat) {
			params.fractal.0 += 1_u32;
			params.fractal.1 = Expr::from_int(params.fractal.0.clone());
			println!("{} -> {}", params.fractal.0, params.fractal.1.to_string());
			is_redraw_needed = true;
		}
		if window.is_key_pressed_or_down(Key::P, params.keys_repeat) {
			params.fractal.0 -= 1_u32;
			params.fractal.1 = Expr::from_int(params.fractal.0.clone());
			println!("{} -> {}", params.fractal.0, params.fractal.1.to_string());
			is_redraw_needed = true;
		}

		if window.is_key_pressed_or_down(Key::R, params.keys_repeat) {
			zoom  = 1.0;
			cam_x = 0.0;
			cam_y = 0.0;
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
					let n_iters: u32 = params.quality.zoom_to_iters_n(zoom);
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


const I: Complex64 = Complex64::I;


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
	// infinite variants:
	UInt(u64),
	Float(float),
	Complex(Complex64),
	// 1 order
	Neg(Box<Expr>),
	Abs(Box<Expr>),
	Arg(Box<Expr>),
	Exp(Box<Expr>),
	Sqrt(Box<Expr>),
	Sin(Box<Expr>),
	Cos(Box<Expr>),
	Tan(Box<Expr>),
	Sinh(Box<Expr>),
	Cosh(Box<Expr>),
	Tanh(Box<Expr>),
	Ln(Box<Expr>),
	// 2 order
	Sum(Box<(Expr, Expr)>),
	Prod(Box<(Expr, Expr)>),
	Div(Box<(Expr, Expr)>),
	Pow(Box<(Expr, Expr)>),
	// WARNING: IF CHANGED UPDATE CONSTS

	// TODO: I, Conj, Re, Im
}
impl Expr {
	// WARNING: UPDATE HERE IF ENUM CHANGED
	const NUMBER_OF_FINITE_VARIANTS: u64 = 3;
	const NUMBER_OF_INFINITE_VARIANTS: u64 = 3 + 12 + 4;

	fn eval(&self, z: Complex64, prev_z: Complex64, init_z: Complex64) -> Complex64 {
		use Expr::*;
		match &self {
			// 0 order
			Z => z,
			PrevZ => prev_z,
			InitZ => init_z,
			UInt(n) => cfr(*n as float),
			Float(x) => cfr(*x),
			Complex(c) => *c,
			// 1 order
			Neg(e) => -e.eval(z, prev_z, init_z),
			Abs(e) => cfr(e.eval(z, prev_z, init_z).abs()),
			Arg(e) => cfr(e.eval(z, prev_z, init_z).arg()),
			Exp(e) => e.eval(z, prev_z, init_z).exp(),
			Sqrt(e) => e.eval(z, prev_z, init_z).sqrt(),
			Sin(e) => e.eval(z, prev_z, init_z).sin(),
			Cos(e) => e.eval(z, prev_z, init_z).cos(),
			Tan(e) => e.eval(z, prev_z, init_z).tan(),
			Sinh(e) => e.eval(z, prev_z, init_z).sinh(),
			Cosh(e) => e.eval(z, prev_z, init_z).cosh(),
			Tanh(e) => e.eval(z, prev_z, init_z).tanh(),
			Ln(e) => e.eval(z, prev_z, init_z).ln(),
			// 2 order
			Sum(box(l, r)) => l.eval(z, prev_z, init_z) + r.eval(z, prev_z, init_z), // es.into_iter().map(|e| e.eval(z, prev_z, init_z)).sum(),
			Prod(box(l, r)) => l.eval(z, prev_z, init_z) * r.eval(z, prev_z, init_z), // es.into_iter().map(|e| e.eval(z, prev_z, init_z)).product(),
			Div(box(num, denom)) => num.eval(z, prev_z, init_z) / denom.eval(z, prev_z, init_z),
			Pow(box(b, t)) => b.eval(z, prev_z, init_z).powc(t.eval(z, prev_z, init_z)),
			// _ => todo!()
		}
	}

	fn to_string(&self) -> String {
		use Expr::*;
		match self {
			// 0 order
			Z => format!("Z"),
			PrevZ => format!("PrevZ"),
			InitZ => format!("InitZ"),
			UInt(n) => format!("{n}"),
			Float(x) => format!("{x:e}"),
			Complex(z) => format!("{z:e}"),
			// 1 order
			Neg(x) => format!("-({})", x.to_string()),
			Abs(e) => format!("abs({})", e.to_string()),
			Arg(e) => format!("arg({})", e.to_string()),
			Exp(e) => format!("exp({})", e.to_string()),
			Sqrt(e) => format!("sqrt({})", e.to_string()),
			Sin(e) => format!("sin({})", e.to_string()),
			Cos(e) => format!("cos({})", e.to_string()),
			Tan(e) => format!("tan({})", e.to_string()),
			Sinh(e) => format!("sinh({})", e.to_string()),
			Cosh(e) => format!("cosh({})", e.to_string()),
			Tanh(e) => format!("tanh({})", e.to_string()),
			Ln(e) => format!("ln({})", e.to_string()),
			// 2 order
			Sum(box(l, r)) => format!("({}+{})", l.to_string(), r.to_string()), // es.into_iter().map(|e| e.to_string()).intersperse(format!("+")).collect(),
			Prod(box(l, r)) => format!("({}*{})", l.to_string(), r.to_string()), // es.into_iter().map(|e| e.to_string()).intersperse(format!("*")).collect(),
			Div(box(num, denom)) => format!("({})/({})", num.to_string(), denom.to_string()),
			Pow(box(b, t)) => format!("({})^({})", b.to_string(), t.to_string()),
			// _ => todo!()
		}
	}

	fn new_random() -> Self {
		let id = todo!();
		Self::from_int(id)
	}

	fn from_u64(id: u64) -> Self {
		Self::from_int(BigUint::from(id))
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
	fn from_int(id: BigUint) -> Self {
		use Expr::*;
		// dbg!(&id);
		if id < BigUint::from(Self::NUMBER_OF_FINITE_VARIANTS) {
			let id: u32 = *id.to_u32_digits().get(0).unwrap_or(&0);
			match id {
				0 => Z,
				1 => PrevZ,
				2 => InitZ,
				_ => unreachable!()
			}
		}
		else {
			let id = id - Self::NUMBER_OF_FINITE_VARIANTS;
			let id_inner = &id / Self::NUMBER_OF_INFINITE_VARIANTS;
			let id_inner_u64: u64 = *id_inner.to_u64_digits().get(0).unwrap_or(&0);
			// todo!("id=id-N or (id-N)%...?");
			let id_mod: u32 = *(id % Self::NUMBER_OF_INFINITE_VARIANTS).to_u32_digits().get(0).unwrap_or(&0);
			match id_mod {
				0 => UInt(id_inner_u64),
				1 => Float(float::from_bits(id_inner_u64)),
				2 => {
					let (k, l) = snake_split_2d_u64(id_inner_u64);
					let u = f32::from_bits(k as u32) as float;
					let v = f32::from_bits(l as u32) as float;
					Complex(Complex64::new(u, v))
				}
				// 1 order
				3 => Neg(bx(Self::from_int(id_inner))),
				4 => Abs(bx(Self::from_int(id_inner))),
				5 => Arg(bx(Self::from_int(id_inner))),
				6 => Exp(bx(Self::from_int(id_inner))),
				7 => Sqrt(bx(Self::from_int(id_inner))),
				8 => Sin(bx(Self::from_int(id_inner))),
				9 => Cos(bx(Self::from_int(id_inner))),
				10=> Tan(bx(Self::from_int(id_inner))),
				11 => Sinh(bx(Self::from_int(id_inner))),
				12 => Cosh(bx(Self::from_int(id_inner))),
				13 => Tanh(bx(Self::from_int(id_inner))),
				14 => Ln(bx(Self::from_int(id_inner))),
				// 2 order
				15 => {
					let (k, l) = snake_split_2d(id_inner);
					let u = Self::from_int(k);
					let v = Self::from_int(l);
					Sum(bx((u, v)))
				}
				16 => {
					let (k, l) = snake_split_2d(id_inner);
					let u = Self::from_int(k);
					let v = Self::from_int(l);
					Prod(bx((u, v)))
				}
				17 => {
					let (k, l) = snake_split_2d(id_inner);
					let u = Self::from_int(k);
					let v = Self::from_int(l);
					Div(bx((u, v)))
				}
				18 => {
					let (k, l) = snake_split_2d(id_inner);
					let u = Self::from_int(k);
					let v = Self::from_int(l);
					Pow(bx((u, v)))
				}
				_ => unreachable!()
			}
		}
	}

	fn to_int(&self) -> u64 {
		todo!()
	}

	fn contains_z(&self) -> bool {
		use Expr::*;
		match self {
			Z => true,
			PrevZ => false,
			InitZ => false,
			UInt(_n) => false,
			Float(_x) => false,
			Complex(_c) => false,
			// 1 order
			Neg(e) => e.contains_z(),
			Abs(e) => e.contains_z(),
			Arg(e) => e.contains_z(),
			Exp(e) => e.contains_z(),
			Sqrt(e) => e.contains_z(),
			Sin(e) => e.contains_z(),
			Cos(e) => e.contains_z(),
			Tan(e) => e.contains_z(),
			Sinh(e) => e.contains_z(),
			Cosh(e) => e.contains_z(),
			Tanh(e) => e.contains_z(),
			Ln(e) => e.contains_z(),
			// 2 order
			Sum(box(l, r)) => l.contains_z() || r.contains_z(),
			Prod(box(l, r)) => l.contains_z() || r.contains_z(),
			Div(box(num, denom)) => num.contains_z() || denom.contains_z(),
			Pow(box(b, t)) => b.contains_z() || t.contains_z(),
		}
	}
}

fn snake_split_2d(n: BigUint) -> (BigUint, BigUint) {
	if n == BigUint::ZERO { return (BigUint::ZERO, BigUint::ZERO) }
	/*
	0  2  5  9  14 20 27 35
	1  4  8  13 19 26 34
	3  7  12 18 25 33
	6  11 17 24 32
	10 16 23 31
	15 22 30
	21 29
	28
	*/
	let mut row_i = BigUint::ZERO;
	let mut sum = BigUint::ZERO;
	while &sum + &row_i < n {
		row_i += BigUint::one();
		sum += &row_i;
	}
	// dbg!(row_i, sum);
	let delta = n - sum;
	let k = delta.clone();
	let l = row_i - delta;
	(k, l)
}

fn snake_split_2d_u64(n: u64) -> (u64, u64) {
	let (k, l) = snake_split_2d(BigUint::from(n));
	let k = *k.to_u64_digits().get(0).unwrap_or(&0);
	let l = *l.to_u64_digits().get(0).unwrap_or(&0);
	(k, l)
}

#[cfg(test)]
mod snake_split_2d_u64 {
	use super::*;

	#[test] fn _0() { assert_eq!(snake_split_2d_u64(0), (0, 0)) }

	#[test] fn _1() { assert_eq!(snake_split_2d_u64(1), (0, 1)) }
	#[test] fn _2() { assert_eq!(snake_split_2d_u64(2), (1, 0)) }

	#[test] fn _3() { assert_eq!(snake_split_2d_u64(3), (0, 2)) }
	#[test] fn _4() { assert_eq!(snake_split_2d_u64(4), (1, 1)) }
	#[test] fn _5() { assert_eq!(snake_split_2d_u64(5), (2, 0)) }

	#[test] fn _6() { assert_eq!(snake_split_2d_u64(6), (0, 3)) }
	#[test] fn _7() { assert_eq!(snake_split_2d_u64(7), (1, 2)) }
	#[test] fn _8() { assert_eq!(snake_split_2d_u64(8), (2, 1)) }
	#[test] fn _9() { assert_eq!(snake_split_2d_u64(9), (3, 0)) }

	#[test] fn _10() { assert_eq!(snake_split_2d_u64(10), (0, 4)) }
	#[test] fn _11() { assert_eq!(snake_split_2d_u64(11), (1, 3)) }
	#[test] fn _12() { assert_eq!(snake_split_2d_u64(12), (2, 2)) }
	#[test] fn _13() { assert_eq!(snake_split_2d_u64(13), (3, 1)) }
	#[test] fn _14() { assert_eq!(snake_split_2d_u64(14), (4, 0)) }

	#[test] fn _15() { assert_eq!(snake_split_2d_u64(15), (0, 5)) }
	#[test] fn _16() { assert_eq!(snake_split_2d_u64(16), (1, 4)) }
	#[test] fn _17() { assert_eq!(snake_split_2d_u64(17), (2, 3)) }
	#[test] fn _18() { assert_eq!(snake_split_2d_u64(18), (3, 2)) }
	#[test] fn _19() { assert_eq!(snake_split_2d_u64(19), (4, 1)) }
	#[test] fn _20() { assert_eq!(snake_split_2d_u64(20), (5, 0)) }
}



#[cfg(test)]
mod expr_from_int {
	use super::*;
	use Expr::*;
	#[test] fn _0() { assert_eq!(Expr::from_u64(0), Z) }
	#[test] fn _1() { assert_eq!(Expr::from_u64(1), PrevZ) }
	#[test] fn _2() { assert_eq!(Expr::from_u64(2), InitZ) }

	#[test] fn _3() { assert_eq!(Expr::from_u64(3), UInt(0)) }
	#[test] fn _4() { assert_eq!(Expr::from_u64(4), Float(0.)) }
	#[test] fn _5() { assert_eq!(Expr::from_u64(5), Complex(cf(0., 0.))) }
	#[test] fn _6() { assert_eq!(Expr::from_u64(6), Neg(bx(Z))) }
	#[test] fn _7() { assert_eq!(Expr::from_u64(7), Abs(bx(Z))) }
	#[test] fn _8() { assert_eq!(Expr::from_u64(8), Arg(bx(Z))) }
	#[test] fn _9() { assert_eq!(Expr::from_u64(9), Exp(bx(Z))) }
	#[test] fn _10() { assert_eq!(Expr::from_u64(10), Sqrt(bx(Z))) }
	#[test] fn _11() { assert_eq!(Expr::from_u64(11), Sin(bx(Z))) }
	#[test] fn _12() { assert_eq!(Expr::from_u64(12), Cos(bx(Z))) }
	#[test] fn _13() { assert_eq!(Expr::from_u64(13), Tan(bx(Z))) }
	#[test] fn _14() { assert_eq!(Expr::from_u64(14), Sinh(bx(Z))) }
	#[test] fn _15() { assert_eq!(Expr::from_u64(15), Cosh(bx(Z))) }
	#[test] fn _16() { assert_eq!(Expr::from_u64(16), Tanh(bx(Z))) }
	#[test] fn _17() { assert_eq!(Expr::from_u64(17), Ln(bx(Z))) }
	#[test] fn _18() { assert_eq!(Expr::from_u64(18), Sum(bx((Z, Z)))) }
	#[test] fn _19() { assert_eq!(Expr::from_u64(19), Prod(bx((Z, Z)))) }
	#[test] fn _20() { assert_eq!(Expr::from_u64(20), Div(bx((Z, Z)))) }
	#[test] fn _21() { assert_eq!(Expr::from_u64(21), Pow(bx((Z, Z)))) }

	#[test] fn _22() { assert_eq!(Expr::from_u64(22), UInt(1)) }
	#[test] fn _23() { assert_eq!(Expr::from_u64(23), Float(5e-324)) }
	#[test] fn _24() { assert_eq!(Expr::from_u64(24), Complex(cf(0., 1.401298464324817e-45))) }
	#[test] fn _25() { assert_eq!(Expr::from_u64(25), Neg(bx(PrevZ))) }
	#[test] fn _26() { assert_eq!(Expr::from_u64(26), Abs(bx(PrevZ))) }
	#[test] fn _27() { assert_eq!(Expr::from_u64(27), Arg(bx(PrevZ))) }
	#[test] fn _28() { assert_eq!(Expr::from_u64(28), Exp(bx(PrevZ))) }
	#[test] fn _29() { assert_eq!(Expr::from_u64(29), Sqrt(bx(PrevZ))) }
	#[test] fn _30() { assert_eq!(Expr::from_u64(30), Sin(bx(PrevZ))) }
	#[test] fn _31() { assert_eq!(Expr::from_u64(31), Cos(bx(PrevZ))) }
	#[test] fn _32() { assert_eq!(Expr::from_u64(32), Tan(bx(PrevZ))) }
	#[test] fn _33() { assert_eq!(Expr::from_u64(33), Sinh(bx(PrevZ))) }
	#[test] fn _34() { assert_eq!(Expr::from_u64(34), Cosh(bx(PrevZ))) }
	#[test] fn _35() { assert_eq!(Expr::from_u64(35), Tanh(bx(PrevZ))) }
	#[test] fn _36() { assert_eq!(Expr::from_u64(36), Ln(bx(PrevZ))) }
	#[test] fn _37() { assert_eq!(Expr::from_u64(37), Sum(bx((Z, PrevZ)))) }
	#[test] fn _38() { assert_eq!(Expr::from_u64(38), Prod(bx((Z, PrevZ)))) }
	#[test] fn _39() { assert_eq!(Expr::from_u64(39), Div(bx((Z, PrevZ)))) }
	#[test] fn _40() { assert_eq!(Expr::from_u64(40), Pow(bx((Z, PrevZ)))) }

	#[test] fn _41() { assert_eq!(Expr::from_u64(41), UInt(2)) }
	#[test] fn _42() { assert_eq!(Expr::from_u64(42), Float(1e-323)) }
	#[test] fn _43() { assert_eq!(Expr::from_u64(43), Complex(cf(1.401298464324817e-45, 0.))) }
	#[test] fn _44() { assert_eq!(Expr::from_u64(44), Neg(bx(InitZ))) }
	#[test] fn _45() { assert_eq!(Expr::from_u64(45), Abs(bx(InitZ))) }
	#[test] fn _46() { assert_eq!(Expr::from_u64(46), Arg(bx(InitZ))) }
	#[test] fn _47() { assert_eq!(Expr::from_u64(47), Exp(bx(InitZ))) }
	#[test] fn _48() { assert_eq!(Expr::from_u64(48), Sqrt(bx(InitZ))) }
	#[test] fn _49() { assert_eq!(Expr::from_u64(49), Sin(bx(InitZ))) }
	#[test] fn _50() { assert_eq!(Expr::from_u64(50), Cos(bx(InitZ))) }
	#[test] fn _51() { assert_eq!(Expr::from_u64(51), Tan(bx(InitZ))) }
	#[test] fn _52() { assert_eq!(Expr::from_u64(52), Sinh(bx(InitZ))) }
	#[test] fn _53() { assert_eq!(Expr::from_u64(53), Cosh(bx(InitZ))) }
	#[test] fn _54() { assert_eq!(Expr::from_u64(54), Tanh(bx(InitZ))) }
	#[test] fn _55() { assert_eq!(Expr::from_u64(55), Ln(bx(InitZ))) }
	#[test] fn _56() { assert_eq!(Expr::from_u64(56), Sum(bx((PrevZ, Z)))) }
	#[test] fn _57() { assert_eq!(Expr::from_u64(57), Prod(bx((PrevZ, Z)))) }
	#[test] fn _58() { assert_eq!(Expr::from_u64(58), Div(bx((PrevZ, Z)))) }
	#[test] fn _59() { assert_eq!(Expr::from_u64(59), Pow(bx((PrevZ, Z)))) }

	#[test] fn _60() { assert_eq!(Expr::from_u64(60), UInt(3)) }
	#[test] fn _61() { assert_eq!(Expr::from_u64(61), Float(1.5e-323)) }
	#[test] fn _62() { assert_eq!(Expr::from_u64(62), Complex(cf(0., 2.802596928649634e-45))) }
	#[test] fn _63() { assert_eq!(Expr::from_u64(63), Neg(bx(UInt(0)))) }
	#[test] fn _64() { assert_eq!(Expr::from_u64(64), Abs(bx(UInt(0)))) }
	#[test] fn _65() { assert_eq!(Expr::from_u64(65), Arg(bx(UInt(0)))) }
	#[test] fn _66() { assert_eq!(Expr::from_u64(66), Exp(bx(UInt(0)))) }
	#[test] fn _67() { assert_eq!(Expr::from_u64(67), Sqrt(bx(UInt(0)))) }
	#[test] fn _68() { assert_eq!(Expr::from_u64(68), Sin(bx(UInt(0)))) }
	#[test] fn _69() { assert_eq!(Expr::from_u64(69), Cos(bx(UInt(0)))) }
	#[test] fn _70() { assert_eq!(Expr::from_u64(70), Tan(bx(UInt(0)))) }
	#[test] fn _71() { assert_eq!(Expr::from_u64(71), Sinh(bx(UInt(0)))) }
	#[test] fn _72() { assert_eq!(Expr::from_u64(72), Cosh(bx(UInt(0)))) }
	#[test] fn _73() { assert_eq!(Expr::from_u64(73), Tanh(bx(UInt(0)))) }
	#[test] fn _74() { assert_eq!(Expr::from_u64(74), Ln(bx(UInt(0)))) }
	#[test] fn _75() { assert_eq!(Expr::from_u64(75), Sum(bx((Z, InitZ)))) }
	#[test] fn _76() { assert_eq!(Expr::from_u64(76), Prod(bx((Z, InitZ)))) }
	#[test] fn _77() { assert_eq!(Expr::from_u64(77), Div(bx((Z, InitZ)))) }
	#[test] fn _78() { assert_eq!(Expr::from_u64(78), Pow(bx((Z, InitZ)))) }

	/*
	Z,
	PrevZ,
	InitZ,
	// infinite variants:
	Int(i64),
	Float(float),
	Complex(Complex64),
	// 1 order
	Neg(Box<Expr>),
	Abs(Box<Expr>),
	Arg(Box<Expr>),
	Exp(Box<Expr>),
	Sqrt(Box<Expr>),
	Sin(Box<Expr>),
	Cos(Box<Expr>),
	Tan(Box<Expr>),
	Sinh(Box<Expr>),
	Cosh(Box<Expr>),
	Tanh(Box<Expr>),
	Ln(Box<Expr>),
	// 2 order
	Sum(Vec<Expr>),
	Prod(Vec<Expr>),
	Div { num_denom: Box<(Expr, Expr)> },
	Pow { b_t: Box<(Expr, Expr)> },
	*/
}

