use crate::color::Color;
use std::io;
use std::slice::{Iter, IterMut};

pub struct Canvas {
    width: usize,
    height: usize,
    data: Vec<Color>,
}

impl Canvas {
    pub fn new(width: usize, height: usize) -> Canvas {
        Canvas {
            width: width,
            height: height,
            data: vec![Default::default(); width * height],
        }
    }

    pub fn width(&self) -> usize {
        self.width
    }

    pub fn height(&self) -> usize {
        self.height
    }

    pub fn set(&mut self, x: usize, y: usize, u: Color) {
        let i = self.index(x, y);
        self.data[i] = u;
    }

    pub fn get(&self, x: usize, y: usize) -> Color {
        self.data[self.index(x, y)]
    }

    pub fn iter(&self) -> Iter<'_, Color> {
        self.data.iter()
    }

    pub fn iter_mut(&mut self) -> IterMut<'_, Color> {
        self.data.iter_mut()
    }

    pub fn write_ppm<W: io::Write>(&self, w: &mut W) -> Result<(), io::Error> {
        write!(w, "P3\n{} {}\n255\n", self.width, self.height)?;
        for (i, u) in self.iter().enumerate() {
            let [r, g, b] = u.to_bytes();
            let space = if i == 0 { "" } else { " " };
            write!(w, "{}{} {} {}", space, r, g, b)?;
        }
        write!(w, "\n")?;
        Ok(())
    }

    fn index(&self, x: usize, y: usize) -> usize {
        assert!(x < self.width && y < self.height);
        y * self.width + x
    }
}

#[cfg(test)]
mod tests {
    use super::Canvas;
    use crate::color::Color;
    use quickcheck::{Arbitrary, Gen};
    use std::io::BufRead;
    use std::iter;

    #[derive(Clone, Debug)]
    struct Size(usize);

    impl Size {
        const MAX_SIZE: usize = 128;
    }

    impl Arbitrary for Size {
        fn arbitrary(g: &mut Gen) -> Size {
            let size = <usize as Arbitrary>::arbitrary(g) % Size::MAX_SIZE + 1;
            Size(size)
        }
    }

    #[derive(Clone, Debug)]
    struct SizeAndCoord(usize, usize);

    impl Arbitrary for SizeAndCoord {
        fn arbitrary(g: &mut Gen) -> SizeAndCoord {
            let Size(size) = Arbitrary::arbitrary(g);
            let coord = <usize as Arbitrary>::arbitrary(g) % size;
            SizeAndCoord(size, coord)
        }
    }

    #[quickcheck]
    fn new(Size(width): Size, Size(height): Size) -> bool {
        let canvas = Canvas::new(width, height);
        let check = |x, y| canvas.get(x, y).eq_approx(Color::new(0.0, 0.0, 0.0));
        canvas.width == width
            && canvas.height == height
            && (0..width).all(|x| (0..height).all(|y| check(x, y)))
    }

    #[quickcheck]
    fn get_set(Size(width): Size, Size(height): Size, u: Color) -> bool {
        let mut canvas = Canvas::new(width, height);
        let mut check = |x, y| {
            canvas.set(x, y, u);
            canvas.get(x, y).eq_approx(u)
        };
        (0..width).all(|x| (0..height).all(|y| check(x, y)))
    }

    #[quickcheck]
    fn iter_iter_mut(Size(width): Size, Size(height): Size, u: Color) -> bool {
        let mut canvas = Canvas::new(width, height);
        canvas.iter().len() == width * height
            && canvas
                .iter()
                .all(|v| v.eq_approx(Color::new(0.0, 0.0, 0.0)))
            && {
                for v in canvas.iter_mut() {
                    *v = u;
                }
                canvas.iter().all(|v| v.eq_approx(u))
            }
    }

    #[quickcheck]
    fn write_ppm(
        SizeAndCoord(width, x): SizeAndCoord,
        SizeAndCoord(height, y): SizeAndCoord,
        u: Color,
    ) -> bool {
        let mut canvas = Canvas::new(width, height);
        canvas.set(x, y, u);

        let mut buf = vec![];
        canvas.write_ppm(&mut buf).unwrap();

        let expected_data = {
            let i = y * width + x;
            "0 ".repeat(3 * i)
                + {
                    let [r, g, b] = u.to_bytes();
                    &format!("{} {} {}", r, g, b)
                }
                + &" 0".repeat(3 * (width * height - i - 1))
        };

        let mut lines = buf[..].lines();

        lines.next().unwrap().unwrap() == "P3"
            && lines.next().unwrap().unwrap() == format!("{} {}", width, height)
            && lines.next().unwrap().unwrap() == "255"
            && lines.next().unwrap().unwrap() == expected_data
            && *buf.last().unwrap() == 0x0a
    }
}
