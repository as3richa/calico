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
        for y in 0..self.height {
            let mut line_len = 0;

            let mut write = |byte: u8| -> Result<(), io::Error> {
                let len = if byte < 10 {
                    1
                } else if byte < 100 {
                    2
                } else {
                    3
                };

                if line_len + 1 + len > 70 {
                    write!(w, "\n{}", byte)?;
                    line_len = len;
                } else if line_len == 0 {
                    write!(w, "{}", byte)?;
                    line_len = len;
                } else {
                    write!(w, " {}", byte)?;
                    line_len += len + 1;
                }

                Ok(())
            };

            for x in 0..self.width {
                let [r, g, b] = self.data[self.index(x, y)].to_bytes();
                write(r)?;
                write(g)?;
                write(b)?;
            }
            write!(w, "\n")?;
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
            iter::repeat(0u8)
                .take(3 * i)
                .chain(u.to_bytes().iter().map(|&x| x))
                .chain(iter::repeat(0).take(3 * (width * height - i - 1)))
                .collect::<Vec<u8>>()
        };

        let lines = buf[..]
            .lines()
            .map(|line| line.unwrap())
            .collect::<Vec<String>>();

        let data = || {
            lines[3..lines.len() - 1]
                .iter()
                .flat_map(|line| line.split(' ').map(|num| num.parse::<u8>().unwrap()))
                .collect::<Vec<u8>>()
        };

        lines[0] == "P3"
            && lines[1] == format!("{} {}", width, height)
            && lines[2] == "255"
            && data() == expected_data
            && lines[lines.len() - 1] == ""
            && lines.iter().all(|line| line.len() <= 70)
    }
}
