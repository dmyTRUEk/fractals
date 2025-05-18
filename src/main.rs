//! Render random fractals >:3

use std::f64::consts::PI;

use minifb::{Key, Window, WindowOptions};
use num::{complex::{Complex64, ComplexFloat}, Zero};
use rand::{rng, Rng};
use rayon::iter::{IndexedParallelIterator, IntoParallelRefMutIterator, ParallelIterator};

fn main() {
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

	let mut rng = rng();

	let mut frame_i: u64 = 0;

	let mut zoom: float = 1.0;
	let mut cam_x: float = 0.0;
	let mut cam_y: float = 0.0;

	while window.is_open() && !window.is_key_down(Key::Escape) {
		let mut is_redraw_needed: bool = false;

		(w, h) = window.get_size();
		(wf, hf) = (w as float, h as float);
		let new_size = w * h;
		if new_size != buffer.len() {
			buffer.resize(new_size, 0);
			println!("Resized to {w}x{h}");
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
			println!("frame {frame_i}"); frame_i += 1;

			// println!("zoom = {zoom}");
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
					let mut z_last_not_nan = z;
					let mut z_esc = Complex64::zero();
					let mut is_bounded = true;
					let mut escape_iter_n: u32 = 0;
					let n_iters: u32 = 20;
					for i in 0..n_iters {
						if !z.is_nan() {
							z_last_not_nan = z;
						}
						z = z.powi(2) + z_init;
						if z.norm() > 100. || z.is_nan() {
							is_bounded = false;
							escape_iter_n = i;
							z_esc = z;
							// break;
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


const MOVE_STEP: float = 0.1;
const ZOOM_STEP: float = 1.1;


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



#[allow(non_camel_case_types)]
type float = f64;

