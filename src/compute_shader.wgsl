// Input to the shader. The length of the array is determined by what buffer is bound.
//
// Out of bounds accesses.
@group(0) @binding(0)
var<storage, read> input: array<f32>;
// Output of the shader.
@group(0) @binding(1)
var<storage, read_write> output: array<u32>;

// Ideal workgroup size depends on the hardware, the workload, and other factors. However, it should
// _generally_ be a multiple of 64. Common sizes are 64x1x1, 256x1x1; or 8x8x1, 16x16x1 for 2D workloads.
@compute @workgroup_size(64)
fn my_compute_shader(
	@builtin(num_workgroups) num_workgroups: vec3u,
	@builtin(global_invocation_id) global_id: vec3u,
) {
	// While compute invocations are 3d, we're only using one dimension.
	//let index = global_id.x;

	// Because we're using a workgroup size of 64, if the input size isn't a multiple of 64,
	// we will have some "extra" invocations. This is fine, but we should tell them to stop
	// to avoid out-of-bounds accesses.
	//let array_length = arrayLength(&input);
	//if (global_id.x >= array_length) {
	//	return;
	//}

	// let x = input[global_id.x];
	let x = f32(global_id.x);
	let y = f32(global_id.y);
	let wf = f32(num_workgroups.x);
	let hf = f32(num_workgroups.y);
	let cam_x   = input[0];
	let cam_y   = input[1];
	let zoom    = input[2];
	let n_iters = u32(input[3]);
	let break_loop = input[4] > 0.5;
	let assign_zesc_once = input[5] > 0.5;
	let zesc_value = input[6];
	let alpha = input[7];

	let zesc_value_sqr = pow(zesc_value, 2);

	let _z_init = screen_to_world(x, y, wf, hf, cam_x, cam_y, zoom);
	let z_init = C(_z_init.x, _z_init.y);
	let color = get_fractal_color(
		z_init,
		n_iters,
		break_loop,
		assign_zesc_once,
		zesc_value_sqr,
		alpha,
	);

	let index = u32(round(y*wf + x));
	output[index] = color;
}



fn get_fractal_color(
	z_init: C,
	n_iters: u32,
	break_loop: bool,
	assign_zesc_once: bool,
	zesc_value_sqr: f32,
	alpha: f32,
) -> u32 {
	var z = z_init;
	var z_prev = z_init;
	var z_esc = z_init;
	var z_last_not_nan = z_init;
	var is_bounded = true;
	for (var j: u32 = 0; j < n_iters; j++) {
		if !cis_nan(z) {
			z_last_not_nan = z;
		}
		//z = cadd(cmul(z, z), z_init);
		let z_new = REPLACEME;
		z_prev = z;
		z = z_new;
		let z_check = cnorm_sqr(z) > zesc_value_sqr || cis_nan(z);
		//let is_bounded_check = if assign_zesc_once { is_bounded } else { true };
		let is_bounded_check = (!assign_zesc_once) || is_bounded;
		if z_check && is_bounded_check {
			is_bounded = false;
			//escape_iter_n = j;
			z_esc = z;
			if break_loop {
				break;
			}
		}
	}
	var color = BLACK;
	if !is_bounded {
		// let t = (escape_iter_n as float) + 1. - z.abs().ln().ln() / (2.).ln();
		// let t = ((escape_iter_n + 1) as float) - z.abs().ln().log2();
		//assert!(!z_last_not_nan.is_nan(), "{z_last_not_nan}");
		var z: C;
		if !cis_nan(z_esc) {
			z = z_esc;
		} else {
			z = z_last_not_nan;
		}
		let t = cnorm(z);
		color = value_to_color(t);
	};
	return color;
}

fn screen_to_world(x: f32, y: f32, wf: f32, hf: f32, cam_x: f32, cam_y: f32, zoom: f32) -> vec2f {
	let aspect_ratio = hf / wf;
	let world_x = cam_x + (x - wf * 0.5) / (wf * 0.5) / zoom;
	let world_y = cam_y + (y - hf * 0.5) / (hf * 0.5) / zoom * aspect_ratio;
	return vec2(world_x, world_y);
}



const WHITE: u32 = 0xffffff;
const BLACK: u32 = 0x000000;

const RED    : u32 = 0xff0000;
const ORANGE : u32 = 0xff8800;
const YELLOW : u32 = 0xffff00;
const YELLEN : u32 = 0x88ff00;
const GREEN  : u32 = 0x00ff00;
const GRYAN  : u32 = 0x00ff88;
const CYAN   : u32 = 0x00ffff;
const BLYUAN : u32 = 0x0088ff;
const BLUE   : u32 = 0x0000ff;
const BRED   : u32 = 0x8800ff;
const MAGENTA: u32 = 0xff00ff;
const REGENTA: u32 = 0xff0088;


fn u32_to_u8x4(x: u32) -> vec4u {
	return vec4u(
		(x >> 0u)  & 0xFFu,  // lowest byte
		(x >> 8u)  & 0xFFu,
		(x >> 16u) & 0xFFu,
		(x >> 24u) & 0xFFu   // highest byte
	);
}

fn u8x4_to_u32(bytes: vec4u) -> u32 {
	return
		(bytes.x << 0u) |
		(bytes.y << 8u) |
		(bytes.z << 16u) |
		(bytes.w << 24u);
}

fn lerp(t: f32, p: u32, q: u32) -> u32 {
	let p_argb = u32_to_u8x4(p);
	let pa = p_argb[0];
	let pr = p_argb[1];
	let pg = p_argb[2];
	let pb = p_argb[3];
	let q_argb = u32_to_u8x4(q);
	let qa = q_argb[0];
	let qr = q_argb[1];
	let qg = q_argb[2];
	let qb = q_argb[3];
	return u8x4_to_u32(vec4(
		u32( f32(pa) * (1.-t) + f32(qa) * t ),
		u32( f32(pr) * (1.-t) + f32(qr) * t ),
		u32( f32(pg) * (1.-t) + f32(qg) * t ),
		u32( f32(pb) * (1.-t) + f32(qb) * t ),
	));
}

const N: u32 = 14;
const rainbow_start_values: array<f32, N> = array(
	-1.01,
	( 0./11.)*2.-1.,
	( 1./11.)*2.-1.,
	( 2./11.)*2.-1.,
	( 3./11.)*2.-1.,
	( 4./11.)*2.-1.,
	( 5./11.)*2.-1.,
	( 6./11.)*2.-1.,
	( 7./11.)*2.-1.,
	( 8./11.)*2.-1.,
	( 9./11.)*2.-1.,
	(10./11.)*2.-1.,
	(11./11.)*2.-1.,
	1.01,
);
const rainbow_colors: array<u32, N> = array(
	WHITE,
	RED,
	ORANGE,
	YELLOW,
	YELLEN,
	GREEN,
	GRYAN,
	CYAN,
	BLYUAN,
	BLUE,
	BRED,
	MAGENTA,
	REGENTA,
	WHITE,
);

/// assumes `points` are sorted by float
fn multipoint_gradient_1d(t: f32, points_start_values: array<f32, N>, points_colors: array<u32, N>) -> u32 {
	if is_nan(t) { return WHITE; }
	var i = 1;
	while points_start_values[i] < t { i += 1; }
	let u = points_start_values[i-1];
	let v = points_start_values[i];
	let color_prev = points_colors[i-1];
	let color_next = points_colors[i];
	let t_new = (t - u) / (v - u);
	return lerp(t_new, color_prev, color_next);
}

fn value_to_color(t_in: f32) -> u32 {
	// let t = ((t.ln() / 1e3).abs().ln() / 1e2).tanh() as f32;
	// 710 = ln(f64::MAX)
	// 89 = ln(f32::MAX)
	let t = tanh((log(t_in) / 89.) * 2. - 1.);

	return multipoint_gradient_1d(t, rainbow_start_values, rainbow_colors);
}



//const f32_nan = 0. / 0.;
const f32_nan_u32 = 0x7FC00000u;
//const f32_nan = bitcast<f32>(f32_nan_u32);
fn f32_nan() -> f32 {
	return bitcast<f32>(f32_nan_u32);
}

fn is_nan(x: f32) -> bool {
	return x != x;
}

//const f32_inf = 1. / 0.;
//const f32_neg_inf = -1. / 0.;
const f32_inf_u32 = 0x7F800000u;
const f32_neg_inf_u32 = 0xFF800000u;

fn f32_inf() -> f32 {
	return bitcast<f32>(f32_inf_u32);
}
fn f32_neg_inf() -> f32 {
	return bitcast<f32>(f32_neg_inf_u32);
}

fn is_inf(x: f32) -> bool {
	//return x == f32_inf || x == f32_neg_inf;
	//return x == 1./0. || x == -1./0.;
	let bits: u32 = bitcast<u32>(x);
	return bits == f32_inf_u32 || bits == f32_neg_inf_u32;
}



struct C {
	re: f32,
	im: f32,
}

const I: C = C(0., 1.);

const Czero = C(0., 0.);
const Cone = C(1., 0.);

fn Cre(x: f32) -> C {
	return C(x, 0.);
}
fn Cim(y: f32) -> C {
	return C(0., y);
}

fn cis_nan(z: C) -> bool {
	return is_nan(z.re) || is_nan(z.im);
}
fn cis_zero(z: C) -> bool {
	return z.re == 0. && z.im == 0.;
}

fn cis_eq(a: C, b: C) -> bool {
	return a.re == b.re && a.im == b.im;
}

fn cneg(z: C) -> C {
	return C(-z.re, -z.im);
}

// from here src is rust's `num` crate.
fn cadd(a: C, b: C) -> C {
	return C(a.re + b.re, a.im + b.im);
}
fn csub(a: C, b: C) -> C {
	return C(a.re - b.re, a.im - b.im);
}
fn cmul(a: C, b: C) -> C {
	return C(
		a.re * b.re - a.im * b.im,
		a.re * b.im + a.im * b.re,
	);
}
fn cmulf(z: C, k: f32) -> C {
	return C(k*z.re, k*z.im);
}
fn csqr(z: C) -> C {
	return cmul(z, z);
}
fn cnorm_sqr(z: C) -> f32 {
	return z.re*z.re + z.im*z.im;
}
fn cnorm(z: C) -> f32 {
	return sqrt(cnorm_sqr(z));
}
fn cnormc(z: C) -> C {
	return Cre(cnorm(z));
}
fn cdiv(a: C, b: C) -> C {
	let norm_sqr = cnorm_sqr(b);
	let re = a.re * b.re + a.im * b.im;
	let im = a.im * b.re - a.re * b.im;
	return C(re / norm_sqr, im / norm_sqr);
}
fn cdivf(z: C, k: f32) -> C {
	return C(z.re/k, z.im/k);
}
fn cpow(a: C, b: C) -> C {
	if cis_zero(b) {
		return Cone;
	}
	return cexp(cmul(b, cln(a)));
}
fn carg(z: C) -> f32 {
	return atan2(z.im, z.re);
}
fn cargc(z: C) -> C {
	return Cre(carg(z));
}
fn cre(z: C) -> f32 {
	return z.re;
}
fn crec(z: C) -> C {
	return Cre(cre(z));
}
fn cim(z: C) -> f32 {
	return z.im;
}
fn cimc(z: C) -> C {
	return Cim(cim(z));
}
fn cconj(z: C) -> C {
	return C(z.re, -z.im);
}
fn cexp(z: C) -> C {
	let re = z.re;
	var im = z.im;
	if is_inf(re) {
		if re < 0. {
			if !is_inf(im) {
				return Czero;
			}
		} else if im == 0. || !is_inf(im) {
			if is_inf(im) {
				im = f32_nan();
			}
			return C(re, im);
		}
	} else if is_nan(re) && im == 0. {
		return z;
	}
	return C_from_polar(exp(re), im);
}
fn cln(z: C) -> C {
	let r_theta = C_to_polar(z);
	let r = r_theta[0];
	let theta = r_theta[1];
	return C(log(r), theta);
}
fn csqrt(z: C) -> C {
	if z.im == 0. {
		if z.re >= 0. {
			// simple positive real √r, and copy `im` for its sign
			return C(sqrt(z.re), z.im);
		} else {
			// √(r e^(iπ)) = √r e^(iπ/2) = i√r
			// √(r e^(-iπ)) = √r e^(-iπ/2) = -i√r
			let re = 0.;
			let im = sqrt(-z.re);
			if im >= 0. {
				return C(re, im);
			} else {
				return C(re, -im);
			}
		}
	} else if z.re == 0. {
		// √(r e^(iπ/2)) = √r e^(iπ/4) = √(r/2) + i√(r/2)
		// √(r e^(-iπ/2)) = √r e^(-iπ/4) = √(r/2) - i√(r/2)
		let x = sqrt(abs(z.im) / 2.);
		if z.im >= 0. {
			return C(x, x);
		} else {
			return C(x, -x);
		}
	} else {
		// formula: sqrt(r e^(it)) = sqrt(r) e^(it/2)
		let r_theta = C_to_polar(z);
		let r = r_theta[0];
		let theta = r_theta[1];
		return C_from_polar(sqrt(r), theta / 2.);
	}
}
fn csin(z: C) -> C {
	return C(
		sin(z.re) * cosh(z.im),
		cos(z.re) * sinh(z.im)
	);
}
fn ccos(z: C) -> C {
	return C(
		cos(z.re) * cosh(z.im),
		-sin(z.re) * sinh(z.im)
	);
}
fn ctan(z: C) -> C {
	let z2 = cadd(z, z);
	return cdivf(C(sin(z2.re), sinh(z2.im)), cos(z2.re) + cosh(z2.im));
}
fn csinh(z: C) -> C {
	return C(
		sinh(z.re) * cos(z.im),
		cosh(z.re) * sin(z.im),
	);
}
fn ccosh(z: C) -> C {
	return C(
		cosh(z.re) * cos(z.im),
		sinh(z.re) * sin(z.im),
	);
}
fn ctanh(z: C) -> C {
	let z2 = cadd(z, z);
	return cdivf(C(sinh(z2.re), sin(z2.im)), cosh(z2.re) + cos(z2.im));
}
fn casin(z: C) -> C {
	return cmul(cneg(I), cln(cadd(csqrt(csub(Cone, csqr(z))), cmul(I, z))));
}
fn cacos(z: C) -> C {
	return cmul(cneg(I), cln(cadd(cmul(I, csqrt(csub(Cone, csqr(z)))), z)));
}
fn catan(z: C) -> C {
	if cis_eq(z, I) {
		return Cim(f32_inf());
	} else if cis_eq(z, cneg(I)) {
		return Cim(f32_neg_inf());
	}
	return cdiv(csub(cln(cadd(Cone, cmul(I, z))), cln(csub(Cone, cmul(I, z)))), cmulf(I, 2.)); // TODO: use `cdivf(... * -i, 2.)`?
}
fn casinh(z: C) -> C {
	return cln(cadd(z, csqrt(cadd(Cone, csqr(z)))));
}
fn cacosh(z: C) -> C {
	return cmulf(cln(cadd(csqrt(cdivf(cadd(z, Cone), 2.)), csqrt(cdivf(csub(z, Cone), 2.)))), 2.);
}
fn catanh(z: C) -> C {
	if cis_eq(z, Cone) {
		return Cre(f32_inf());
	} else if cis_eq(z, cneg(Cone)) {
		return Cre(f32_neg_inf());
	}
	// ((one + self).ln() - (one - self).ln()) / two
	return cdivf(csub(cln(cadd(Cone, z)), cln(csub(Cone, z))), 2.);
}
fn cround(z: C) -> C {
	return C(round(z.re), round(z.im));
}
fn cceil(z: C) -> C {
	return C(ceil(z.re), ceil(z.im));
}
fn cfloor(z: C) -> C {
	return C(floor(z.re), floor(z.im));
}

fn C_from_polar(r: f32, theta: f32) -> C {
	return C(r*cos(theta), r*sin(theta));
}
fn C_to_polar(z: C) -> vec2f {
	return vec2(cnorm(z), carg(z));
}
