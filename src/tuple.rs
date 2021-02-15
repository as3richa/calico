use crate::Float;
use std::ops;

#[cfg(test)]
use crate::eq_approx;

#[derive(Clone, Copy, Debug)]
pub struct Tuple {
    pub x: Float,
    pub y: Float,
    pub z: Float,
    pub w: Float,
}

impl Tuple {
    pub fn new(x: Float, y: Float, z: Float, w: Float) -> Tuple {
        Tuple {
            x: x,
            y: y,
            z: z,
            w: w,
        }
    }

    pub fn point(x: Float, y: Float, z: Float) -> Tuple {
        Tuple::new(x, y, z, 1.0)
    }

    pub fn vector(x: Float, y: Float, z: Float) -> Tuple {
        Tuple::new(x, y, z, 0.0)
    }

    pub fn dot(self, rhs: Tuple) -> Float {
        self.x * rhs.x + self.y * rhs.y + self.z * rhs.z + self.w * rhs.w
    }

    pub fn norm(self) -> Float {
        Float::sqrt(self.dot(self))
    }

    pub fn normalize(self) -> Tuple {
        self / self.norm()
    }

    pub fn cross(self, rhs: Tuple) -> Tuple {
        Tuple::vector(
            self.y * rhs.z - self.z * rhs.y,
            self.z * rhs.x - self.x * rhs.z,
            self.x * rhs.y - self.y * rhs.x,
        )
    }

    pub fn as_tuple3(self) -> Tuple3 {
        Tuple3::new([self.x, self.y, self.z])
    }

    #[cfg(test)]
    pub fn eq_approx(self, rhs: Tuple) -> bool {
        eq_approx(self.x, rhs.x)
            && eq_approx(self.y, rhs.y)
            && eq_approx(self.z, rhs.z)
            && eq_approx(self.w, rhs.w)
    }
}

impl ops::Add<Tuple> for Tuple {
    type Output = Tuple;

    fn add(self, rhs: Tuple) -> Tuple {
        Tuple::new(
            self.x + rhs.x,
            self.y + rhs.y,
            self.z + rhs.z,
            self.w + rhs.w,
        )
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
        Tuple::new(
            self.x - rhs.x,
            self.y - rhs.y,
            self.z - rhs.z,
            self.w - rhs.w,
        )
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
        Tuple::new(self.x * rhs, self.y * rhs, self.z * rhs, self.w * rhs)
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
        Tuple::new(self.x / rhs, self.y / rhs, self.z / rhs, self.w / rhs)
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
        Tuple::new(-self.x, -self.y, -self.z, -self.w)
    }
}

#[derive(Clone, Copy)]
pub struct Tuple3([Float; 3]);

impl Tuple3 {
    pub fn new(v: [Float; 3]) -> Tuple3 {
        Tuple3(v)
    }

    pub fn x(self) -> Float {
        self[0]
    }

    pub fn y(self) -> Float {
        self[1]
    }

    pub fn z(self) -> Float {
        self[2]
    }

    pub fn argmax(self) -> (usize, Float) {
        let max = Float::max(self.x(), Float::max(self.y(), self.z()));
        let axis = if max == self.x() {
            0
        } else if max == self.y() {
            1
        } else {
            2
        };
        (axis, max)
    }

    pub fn dot(self, rhs: Tuple3) -> Float {
        self.x() * rhs.x() + self.y() * rhs.y() + self.z() * rhs.z()
    }

    pub fn cross(self, rhs: Tuple3) -> Tuple3 {
        Tuple3::new([
            self.y() * rhs.z() - self.z() * rhs.y(),
            self.z() * rhs.x() - self.x() * rhs.z(),
            self.x() * rhs.y() - self.y() * rhs.x(),
        ])
    }

    pub fn min(lhs: Tuple3, rhs: Tuple3) -> Tuple3 {
        Tuple3::new([
            Float::min(lhs.x(), rhs.x()),
            Float::min(lhs.y(), rhs.y()),
            Float::min(lhs.z(), rhs.z()),
        ])
    }

    pub fn max(lhs: Tuple3, rhs: Tuple3) -> Tuple3 {
        Tuple3::new([
            Float::max(lhs.x(), rhs.x()),
            Float::max(lhs.y(), rhs.y()),
            Float::max(lhs.z(), rhs.z()),
        ])
    }

    pub fn norm(self) -> Float {
        Float::sqrt(self.dot(self))
    }

    pub fn normalize(self) -> Tuple3 {
        self / self.norm()
    }

    pub fn as_point(self) -> Tuple {
        Tuple::point(self.x(), self.y(), self.z())
    }

    pub fn as_vector(self) -> Tuple {
        Tuple::vector(self.x(), self.y(), self.z())
    }
}

impl ops::Add<Tuple3> for Tuple3 {
    type Output = Tuple3;

    fn add(self, rhs: Tuple3) -> Tuple3 {
        Tuple3::new([self.x() + rhs.x(), self.y() + rhs.y(), self.z() + rhs.z()])
    }
}

impl ops::Sub<Tuple3> for Tuple3 {
    type Output = Tuple3;

    fn sub(self, rhs: Tuple3) -> Tuple3 {
        Tuple3::new([self.x() - rhs.x(), self.y() - rhs.y(), self.z() - rhs.z()])
    }
}

impl ops::Mul<Float> for Tuple3 {
    type Output = Tuple3;

    fn mul(self, rhs: Float) -> Tuple3 {
        Tuple3::new([self.x() * rhs, self.y() * rhs, self.z() * rhs])
    }
}

impl ops::Div<Float> for Tuple3 {
    type Output = Tuple3;

    fn div(self, rhs: Float) -> Tuple3 {
        Tuple3::new([self.x() / rhs, self.y() / rhs, self.z() / rhs])
    }
}

impl ops::Index<usize> for Tuple3 {
    type Output = Float;

    fn index(&self, i: usize) -> &Float {
        &self.0[i]
    }
}

#[derive(Clone, Copy)]
pub struct Tuple2 {
    x: Float,
    y: Float,
}

impl Tuple2 {
    pub fn new(x: Float, y: Float) -> Tuple2 {
        Tuple2 { x: x, y: y }
    }
}

#[cfg(test)]
pub mod tests {
    use super::Tuple;
    use crate::finite::Finite;
    use crate::{eq_approx, Float, EPSILON};
    use quickcheck::{Arbitrary, Gen, TestResult};

    impl Arbitrary for Tuple {
        fn arbitrary(gen: &mut Gen) -> Tuple {
            let Finite(x) = Arbitrary::arbitrary(gen);
            let Finite(y) = Arbitrary::arbitrary(gen);
            let Finite(z) = Arbitrary::arbitrary(gen);
            let Finite(w) = Arbitrary::arbitrary(gen);
            Tuple::new(x, y, z, w)
        }
    }

    #[derive(Clone, Debug)]
    pub struct Vector(pub Tuple);

    impl Arbitrary for Vector {
        fn arbitrary(gen: &mut Gen) -> Vector {
            let Finite(x) = Arbitrary::arbitrary(gen);
            let Finite(y) = Arbitrary::arbitrary(gen);
            let Finite(z) = Arbitrary::arbitrary(gen);
            Vector(Tuple::vector(x, y, z))
        }
    }

    #[derive(Clone, Debug)]
    pub struct Point(pub Tuple);

    impl Arbitrary for Point {
        fn arbitrary(gen: &mut Gen) -> Point {
            let Finite(x) = Arbitrary::arbitrary(gen);
            let Finite(y) = Arbitrary::arbitrary(gen);
            let Finite(z) = Arbitrary::arbitrary(gen);
            Point(Tuple::point(x, y, z))
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

    #[quickcheck]
    fn add(u: Tuple, v: Tuple) -> bool {
        (u + v).eq_approx(Tuple::new(u.x + v.x, u.y + v.y, u.z + v.z, u.w + v.w))
    }

    #[quickcheck]
    fn add_assign(u: Tuple, v: Tuple) -> bool {
        let mut q = u;
        q += v;
        q.eq_approx(Tuple::new(u.x + v.x, u.y + v.y, u.z + v.z, u.w + v.w))
    }

    #[quickcheck]
    fn sub(u: Tuple, v: Tuple) -> bool {
        (u - v).eq_approx(Tuple::new(u.x - v.x, u.y - v.y, u.z - v.z, u.w - v.w))
    }

    #[quickcheck]
    fn sub_assign(u: Tuple, v: Tuple) -> bool {
        let mut q = u;
        q -= v;
        q.eq_approx(Tuple::new(u.x - v.x, u.y - v.y, u.z - v.z, u.w - v.w))
    }

    #[quickcheck]
    fn mul(u: Tuple, Finite(a): Finite) -> bool {
        (u * a).eq_approx(Tuple::new(a * u.x, a * u.y, a * u.z, a * u.w))
    }

    #[quickcheck]
    fn mul_assign(u: Tuple, Finite(a): Finite) -> bool {
        let mut q = u;
        q *= a;
        q.eq_approx(Tuple::new(a * u.x, a * u.y, a * u.z, a * u.w))
    }

    #[quickcheck]
    fn div(u: Tuple, Finite(a): Finite) -> TestResult {
        if a <= EPSILON {
            TestResult::discard()
        } else {
            TestResult::from_bool((u / a).eq_approx(Tuple::new(u.x / a, u.y / a, u.z / a, u.w / a)))
        }
    }

    #[quickcheck]
    fn div_assign(u: Tuple, Finite(a): Finite) -> TestResult {
        if a <= EPSILON {
            TestResult::discard()
        } else {
            let mut q = u;
            q /= a;
            TestResult::from_bool(q.eq_approx(Tuple::new(u.x / a, u.y / a, u.z / a, u.w / a)))
        }
    }

    #[quickcheck]
    fn neg(u: Tuple) -> bool {
        (-u).eq_approx(Tuple::new(-u.x, -u.y, -u.z, -u.w))
    }

    #[quickcheck]
    fn vector_vector_add(Vector(u): Vector, Vector(v): Vector) -> bool {
        (u + v).eq_approx(Tuple::vector(u.x + v.x, u.y + v.y, u.z + v.z))
    }

    #[quickcheck]
    fn point_vector_add(Point(p): Point, Vector(u): Vector) -> bool {
        (p + u).eq_approx(Tuple::point(p.x + u.x, p.y + u.y, p.z + u.z))
    }

    #[quickcheck]
    fn vector_vector_sub(Vector(u): Vector, Vector(v): Vector) -> bool {
        (u - v).eq_approx(Tuple::vector(u.x - v.x, u.y - v.y, u.z - v.z))
    }

    #[quickcheck]
    fn point_vector_sub(Point(p): Point, Vector(u): Vector) -> bool {
        (p - u).eq_approx(Tuple::point(p.x - u.x, p.y - u.y, p.z - u.z))
    }

    #[quickcheck]
    fn point_point_sub(Point(p): Point, Point(q): Point) -> bool {
        (p - q).eq_approx(Tuple::vector(p.x - q.x, p.y - q.y, p.z - q.z))
    }

    #[quickcheck]
    fn dot(u: Tuple, v: Tuple) -> bool {
        eq_approx(u.dot(v), u.x * v.x + u.y * v.y + u.z * v.z + u.w * v.w)
    }

    #[quickcheck]
    fn norm(u: Tuple) -> bool {
        eq_approx(u.norm(), Float::sqrt(u.dot(u)))
    }

    #[quickcheck]
    fn normalize(u: Tuple) -> TestResult {
        if eq_approx(u.norm(), 0.0) {
            TestResult::discard()
        } else {
            let v = u.normalize();
            TestResult::from_bool(eq_approx(v.norm(), 1.0) && u.eq_approx(v * u.norm()))
        }
    }

    #[quickcheck]
    fn cross() -> bool {
        let u = Tuple::vector(1.0, 0.0, 0.0).cross(Tuple::vector(0.0, 1.0, 0.0));
        let v = Tuple::vector(0.0, 1.0, 0.0).cross(Tuple::vector(1.0, 0.0, 0.0));
        u.eq_approx(Tuple::vector(0.0, 0.0, 1.0)) && v.eq_approx(Tuple::vector(0.0, 0.0, -1.0))
    }

    #[quickcheck]
    fn cross_neg(Vector(u): Vector, Vector(v): Vector) -> bool {
        u.cross(v).eq_approx(-v.cross(u))
    }
}
