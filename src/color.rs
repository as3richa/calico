use crate::Float;
use std::ops;

#[cfg(test)]
use crate::eq_approx;

#[derive(Clone, Copy, Debug, Default)]
pub struct Color {
    r: Float,
    g: Float,
    b: Float,
}

impl Color {
    pub const fn new(r: Float, g: Float, b: Float) -> Color {
        Color { r: r, g: g, b: b }
    }

    pub fn from_bytes(r: u8, g: u8, b: u8) -> Color {
        fn to_float(x: u8) -> Float {
            (x as Float) / 255.0
        }
        Color::new(to_float(r), to_float(g), to_float(b))
    }

    pub fn to_bytes(self) -> [u8; 3] {
        fn to_byte(x: Float) -> u8 {
            let clamped = Float::max(Float::min(x, 1.0), 0.0) * 255.0;
            Float::round(clamped) as u8
        }
        [to_byte(self.r), to_byte(self.g), to_byte(self.b)]
    }

    pub fn blend(self, rhs: Color) -> Color {
        Color::new(self.r * rhs.r, self.g * rhs.g, self.b * rhs.b)
    }

    #[cfg(test)]
    pub fn eq_approx(self, rhs: Color) -> bool {
        eq_approx(self.r, rhs.r) && eq_approx(self.g, rhs.g) && eq_approx(self.b, rhs.b)
    }
}

impl ops::Add<Color> for Color {
    type Output = Color;

    fn add(self, rhs: Color) -> Color {
        Color::new(self.r + rhs.r, self.g + rhs.g, self.b + rhs.b)
    }
}

impl ops::AddAssign<Color> for Color {
    fn add_assign(&mut self, rhs: Color) {
        self.r += rhs.r;
        self.g += rhs.g;
        self.b += rhs.b;
    }
}

impl ops::Sub<Color> for Color {
    type Output = Color;

    fn sub(self, rhs: Color) -> Color {
        Color::new(self.r - rhs.r, self.g - rhs.g, self.b - rhs.b)
    }
}

impl ops::SubAssign<Color> for Color {
    fn sub_assign(&mut self, rhs: Color) {
        self.r -= rhs.r;
        self.g -= rhs.g;
        self.b -= rhs.b;
    }
}

impl ops::Mul<Float> for Color {
    type Output = Color;

    fn mul(self, rhs: Float) -> Color {
        Color::new(self.r * rhs, self.g * rhs, self.b * rhs)
    }
}

impl ops::MulAssign<Float> for Color {
    fn mul_assign(&mut self, rhs: Float) {
        self.r *= rhs;
        self.g *= rhs;
        self.b *= rhs;
    }
}

#[cfg(test)]
mod tests {
    use super::Color;
    use crate::finite::Finite;
    use crate::Float;
    use quickcheck::{Arbitrary, Gen};

    impl Arbitrary for Color {
        fn arbitrary(gen: &mut Gen) -> Color {
            let Finite(r) = Arbitrary::arbitrary(gen);
            let Finite(g) = Arbitrary::arbitrary(gen);
            let Finite(b) = Arbitrary::arbitrary(gen);
            Color::new(r, g, b)
        }
    }

    #[quickcheck]
    fn default() -> bool {
        let x: Color = Default::default();
        x.eq_approx(Color::new(0.0, 0.0, 0.0))
    }

    #[quickcheck]
    fn add(u: Color, v: Color) -> bool {
        (u + v).eq_approx(Color::new(u.r + v.r, u.g + v.g, u.b + v.b))
    }

    #[quickcheck]
    fn add_assign(u: Color, v: Color) -> bool {
        let mut q = u;
        q += v;
        q.eq_approx(Color::new(u.r + v.r, u.g + v.g, u.b + v.b))
    }

    #[quickcheck]
    fn sub(u: Color, v: Color) -> bool {
        (u - v).eq_approx(Color::new(u.r - v.r, u.g - v.g, u.b - v.b))
    }

    #[quickcheck]
    fn sub_assign(u: Color, v: Color) -> bool {
        let mut q = u;
        q -= v;
        q.eq_approx(Color::new(u.r - v.r, u.g - v.g, u.b - v.b))
    }

    #[quickcheck]
    fn mul(u: Color, Finite(a): Finite) -> bool {
        (u * a).eq_approx(Color::new(a * u.r, a * u.g, a * u.b))
    }

    #[quickcheck]
    fn mul_assign(u: Color, Finite(a): Finite) -> bool {
        let mut q = u;
        q *= a;
        q.eq_approx(Color::new(a * u.r, a * u.g, a * u.b))
    }

    #[quickcheck]
    fn bytes_roundtrip(r: u8, g: u8, b: u8) -> bool {
        Color::from_bytes(r, g, b).to_bytes() == [r, g, b]
    }

    #[quickcheck]
    fn to_bytes(Finite(r): Finite, Finite(g): Finite, Finite(b): Finite) -> bool {
        fn check(x: Float, x2: u8) -> bool {
            if x < 0.0 {
                x2 == 0
            } else if x > 1.0 {
                x2 == 255
            } else {
                x2 == (255.0 * x.round()) as u8
            }
        }
        let [r2, g2, b2] = Color::new(r, g, b).to_bytes();
        check(r, r2) && check(g, g2) && check(b, b2)
    }

    #[quickcheck]
    fn blend(u: Color, v: Color) -> bool {
        u.blend(v)
            .eq_approx(Color::new(u.r * v.r, u.g * v.g, u.b * v.b))
    }
}
