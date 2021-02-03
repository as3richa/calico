mod tuple {
    use crate::{Float, EPSILON};
    use std::ops;

    struct Tuple {
        x: Float,
        y: Float,
        z: Float,
        w: Float,
    }

    impl Tuple {
        fn new(x: Float, y: Float, z: Float, w: Float) -> Tuple {
            Tuple {
                x: x,
                y: y,
                z: z,
                w: w,
            }
        }
        fn point(x: Float, y: Float, z: Float) -> Tuple {
            Tuple::new(x, y, z, 1.0)
        }

        fn vector(x: Float, y: Float, z: Float) -> Tuple {
            Tuple::new(x, y, z, 0.0)
        }

        fn eq_approx(self, rhs: Tuple) -> bool {
            self.eq_approx_eps(rhs, EPSILON)
        }

        fn eq_approx_eps(self, rhs: Tuple, epsilon: Float) -> bool {
            Float::abs(self.x - rhs.x) < epsilon
                && Float::abs(self.y - rhs.y) < epsilon
                && Float::abs(self.z - rhs.z) < epsilon
                && Float::abs(self.w - rhs.w) < epsilon
        }
    }

    impl ops::Add<Tuple> for Tuple {
        type Output = Tuple;

        fn add(self, rhs: Tuple) -> Tuple {
            Tuple {
                x: self.x + rhs.x,
                y: self.y + rhs.y,
                z: self.z + rhs.z,
                w: self.w + rhs.w,
            }
        }
    }

    impl ops::AddAssign<Tuple> for Tuple {
        fn add_assign(&mut self, rhs: Tuple) {
            self.x += rhs.x;
            self.y += rhs.y;
            self.z += rhs.z;
            self.w += rhs.w;
        }
    }

    impl ops::Sub<Tuple> for Tuple {
        type Output = Tuple;

        fn sub(self, rhs: Tuple) -> Tuple {
            Tuple {
                x: self.x - rhs.x,
                y: self.y - rhs.y,
                z: self.z - rhs.z,
                w: self.w - rhs.w,
            }
        }
    }

    impl ops::SubAssign<Tuple> for Tuple {
        fn sub_assign(&mut self, rhs: Tuple) {
            self.x -= rhs.x;
            self.y -= rhs.y;
            self.z -= rhs.z;
            self.w -= rhs.w;
        }
    }

    impl ops::Mul<Float> for Tuple {
        type Output = Tuple;

        fn mul(self, rhs: Float) -> Tuple {
            Tuple {
                x: rhs * self.x,
                y: rhs * self.y,
                z: rhs * self.z,
                w: rhs * self.w,
            }
        }
    }

    impl ops::MulAssign<Float> for Tuple {
        fn mul_assign(&mut self, rhs: Float) {
            self.x *= rhs;
            self.y *= rhs;
            self.z *= rhs;
            self.w *= rhs;
        }
    }

    impl ops::Div<Float> for Tuple {
        type Output = Tuple;

        fn div(self, rhs: Float) -> Tuple {
            Tuple {
                x: self.x / rhs,
                y: self.y / rhs,
                z: self.z / rhs,
                w: self.w / rhs,
            }
        }
    }

    impl ops::DivAssign<Float> for Tuple {
        fn div_assign(&mut self, rhs: Float) {
            self.x /= rhs;
            self.y /= rhs;
            self.z /= rhs;
            self.w /= rhs;
        }
    }

    impl ops::Neg for Tuple {
        type Output = Self;

        fn neg(self) -> Tuple {
            Tuple {
                x: -self.x,
                y: -self.y,
                z: -self.z,
                w: -self.w,
            }
        }
    }

    #[cfg(test)]
    mod tests {
        use super::{Float, Tuple};
        use quickcheck::{Arbitrary, Gen};
        use std::fmt::{Debug, Error, Formatter};

        #[derive(Clone)]
        struct Finite(Float);

        impl Finite {
            const MAX_SIZE: Float = 1e12;
        }

        impl Debug for Finite {
            fn fmt(&self, f: &mut Formatter) -> Result<(), Error> {
                <Float as Debug>::fmt(&self.0, f)
            }
        }

        impl Arbitrary for Finite {
            fn arbitrary(g: &mut Gen) -> Finite {
                loop {
                    let x = <Float as Arbitrary>::arbitrary(g);

                    if !x.is_nan() && x.abs() <= Finite::MAX_SIZE {
                        return Finite(x);
                    }
                }
            }
        }

        #[quickcheck]
        fn point((Finite(x), Finite(y), Finite(z)): (Finite, Finite, Finite)) -> bool {
            Tuple::point(x, y, z).eq_approx(Tuple::new(x, y, z, 1.0))
        }

        #[quickcheck]
        fn vector((Finite(x), Finite(y), Finite(z)): (Finite, Finite, Finite)) -> bool {
            Tuple::vector(x, y, z).eq_approx(Tuple::new(x, y, z, 0.0))
        }
    }
}
