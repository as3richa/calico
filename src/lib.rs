#[cfg(test)]
extern crate quickcheck;

#[cfg(test)]
#[macro_use(quickcheck)]
extern crate quickcheck_macros;

type Float = f32;
const EPSILON: Float = 1e-5;

mod tuple;
