//! Render random fractals >:3

use minifb::{Key, Window, WindowOptions};

fn main() {
	let mut width  = 320;
	let mut height = 240;
	let mut buffer: Vec<u32> = vec![0; width * height];

	let mut window = Window::new(
		"fractals",
		width,
		height,
		WindowOptions {
			resize: true,
			..WindowOptions::default()
		},
	).unwrap();

	// Limit to max ~60 fps update rate
	window.set_target_fps(60);

	while window.is_open() && !window.is_key_down(Key::Escape) {
		(width, height) = window.get_size();
		let new_size = width * height;
		if new_size != buffer.len() {
			buffer.resize(new_size, 0);
			println!("Resized to {width}x{height}");
		}

		for (i, pixel) in buffer.iter_mut().enumerate() {
			*pixel += i as u32;
		}

		window.update_with_buffer(&buffer, width, height).unwrap();
	}
}

