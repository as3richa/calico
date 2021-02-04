use crate::Float;
use quickcheck::{Arbitrary, Gen};
use std::fmt::Debug;

#[derive(Clone, Debug)]
pub struct Finite(pub Float);

impl Finite {
    const MAX_SIZE: Float = 1e12;
}

impl Arbitrary for Finite {
    fn arbitrary(g: &mut Gen) -> Finite {
        loop {
            let x: Float = Arbitrary::arbitrary(g);

            if !x.is_nan() && x.abs() <= Finite::MAX_SIZE {
                return Finite(x);
            }
        }
    }
}
