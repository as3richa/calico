#[cfg(test)]
extern crate quickcheck;

#[cfg(test)]
#[macro_use(quickcheck)]
extern crate quickcheck_macros;

pub type Float = f32;
pub use std::f32::consts;

#[cfg(test)]
const EPSILON: Float = 1e-4;

#[cfg(test)]
fn eq_approx(x: Float, y: Float) -> bool {
    Float::abs(x - y) < EPSILON
        || Float::abs(x - y) / Float::max(Float::abs(x), Float::abs(y)) < EPSILON
}

mod canvas;
mod color;
mod matrix;
mod tuple;

#[cfg(test)]
mod finite;

pub use canvas::Canvas;
pub use color::Color;
pub use matrix::Matrix;
pub use tuple::Tuple;
