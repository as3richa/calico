use crate::Float;
use std::ops;

#[cfg(test)]
use crate::{eq_approx_eps, EPSILON};

#[derive(Clone, Copy, Debug)]
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

    fn dot(self, rhs: Tuple) -> Float {
        self.x * rhs.x + self.y * rhs.y + self.z * rhs.z + self.w * rhs.w
    }

    fn norm_sq(self) -> Float {
        self.dot(self)
    }

    fn norm(self) -> Float {
        Float::sqrt(self.norm_sq())
    }

    fn normalize(self) -> Tuple {
        self / self.norm()
    }

    fn cross(self, rhs: Tuple) -> Tuple {
        Tuple::vector(
            self.y * rhs.z - self.z * rhs.y,
            self.z * rhs.x - self.x * rhs.z,
            self.x * rhs.y - self.y * rhs.x,
        )
    }

    #[cfg(test)]
    fn eq_approx(self, rhs: Tuple) -> bool {
        self.eq_approx_eps(rhs, EPSILON)
    }

    #[cfg(test)]
    fn eq_approx_eps(self, rhs: Tuple, epsilon: Float) -> bool {
        eq_approx_eps(self.x, rhs.x, epsilon)
            && eq_approx_eps(self.y, rhs.y, epsilon)
            && eq_approx_eps(self.z, rhs.z, epsilon)
            && eq_approx_eps(self.w, rhs.w, epsilon)
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

#[cfg(test)]
mod tests {
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
    struct Vector(Tuple);

    impl Arbitrary for Vector {
        fn arbitrary(gen: &mut Gen) -> Vector {
            let Finite(x) = Arbitrary::arbitrary(gen);
            let Finite(y) = Arbitrary::arbitrary(gen);
            let Finite(z) = Arbitrary::arbitrary(gen);
            Vector(Tuple::vector(x, y, z))
        }
    }

    #[derive(Clone, Debug)]
    struct Point(Tuple);

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
        eq_approx(u.norm(), Float::sqrt(u.dot(u))) && eq_approx(u.norm_sq(), u.norm() * u.norm())
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
    fn cross(Vector(u): Vector, Vector(v): Vector) -> bool {
        let s = u.cross(v);
        if eq_approx(u.norm(), 0.0) || eq_approx(v.norm(), 0.0) {
            eq_approx(s.norm(), 0.0)
        } else {
            let sn = Float::max(s.norm(), 1.0);
            eq_approx(s.dot(u) / (u.norm() * sn), 0.0) && eq_approx(s.dot(v) / (v.norm() * sn), 0.0)
        }
    }

    #[quickcheck]
    fn cross_neg(Vector(u): Vector, Vector(v): Vector) -> bool {
        u.cross(v).eq_approx(-v.cross(u))
    }

    #[quickcheck]
    fn cross_axes() -> bool {
        Tuple::vector(1.0, 0.0, 0.0)
            .cross(Tuple::vector(0.0, 1.0, 0.0))
            .eq_approx(Tuple::vector(0.0, 0.0, 1.0))
    }

    #[quickcheck]
    fn cross_distributive(Vector(u): Vector, Vector(v): Vector, Vector(q): Vector) -> bool {
        u.cross(v + q).eq_approx(u.cross(v) + u.cross(q))
    }

    #[quickcheck]
    fn simulate_projectile() -> bool {
        let initial_pos = Tuple::point(0.0, 100.0, 0.0);
        let initial_vel = Tuple::vector(10.0, 0.0, 0.0);
        let gravity = Tuple::vector(0.0, -9.8, 0.0);
        let total_time = Float::sqrt(initial_pos.y / (-gravity.y / 2.0));

        let steps = 10000;
        let step_time = total_time / (steps as Float);

        let mut pos = initial_pos;

        for step in 0..steps {
            let vel = initial_vel + gravity * (step as Float) / (steps as Float) * total_time;
            pos += vel * step_time;
        }

        pos.eq_approx_eps(Tuple::point(initial_vel.x * total_time, 0.0, 0.0), 1e-2)
    }
}
