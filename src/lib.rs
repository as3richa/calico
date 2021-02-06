#[cfg(test)]
extern crate quickcheck;

#[cfg(test)]
#[macro_use(quickcheck)]
extern crate quickcheck_macros;

pub type Float = f32;

#[cfg(test)]
const EPSILON: Float = 1e-4;

#[cfg(test)]
fn eq_approx(x: Float, y: Float) -> bool {
    eq_approx_eps(x, y, EPSILON)
}

#[cfg(test)]
fn eq_approx_eps(x: Float, y: Float, epsilon: Float) -> bool {
    Float::abs(x - y) < epsilon
        || Float::abs(x - y) / Float::max(Float::abs(x), Float::abs(y)) < epsilon
}

mod canvas;
mod color;
mod matrix;
mod tuple;

#[cfg(test)]
mod finite;

pub use canvas::Canvas;
pub use color::Color;
pub use tuple::Tuple;
