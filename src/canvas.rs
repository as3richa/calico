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
    use std::io::BufRead;

    const SIZES: [(usize, usize); 5] = [(1, 1), (1, 100), (100, 1), (100, 100), (500, 500)];

    fn colors() -> [Color; 10] {
        [
            Color::from_bytes(0, 0, 0),
            Color::from_bytes(255, 255, 255),
            Color::from_bytes(168, 149, 184),
            Color::from_bytes(164, 10, 68),
            Color::from_bytes(254, 181, 174),
            Color::from_bytes(190, 186, 187),
            Color::from_bytes(39, 106, 17),
            Color::from_bytes(61, 178, 147),
            Color::from_bytes(200, 102, 248),
            Color::from_bytes(63, 125, 231),
        ]
    }

    #[test]
    fn new() {
        for &(width, height) in SIZES.iter() {
            let canvas = Canvas::new(width, height);
            let check = |x, y| canvas.get(x, y).eq_approx(Color::new(0.0, 0.0, 0.0));

            assert!(
                canvas.width == width
                    && canvas.height == height
                    && (0..width).all(|x| (0..height).all(|y| check(x, y)))
            );
        }
    }

    #[test]
    fn get_set() {
        for (&(width, height), &u) in SIZES.iter().zip(colors().iter().cycle()) {
            let mut canvas = Canvas::new(width, height);
            for x in 0..width {
                for y in 0..height {
                    canvas.set(x, y, u);
                    assert!(canvas.get(x, y).eq_approx(u));
                }
            }
        }
    }

    #[test]
    fn iter_iter_mut() {
        for (&(width, height), &u) in SIZES.iter().zip(colors().iter().cycle()) {
            let mut canvas = Canvas::new(width, height);
            assert!(canvas.iter().len() == width * height);
            assert!(canvas
                .iter()
                .all(|v| v.eq_approx(Color::new(0.0, 0.0, 0.0))));

            for v in canvas.iter_mut() {
                *v = u;
            }

            assert!(canvas.iter().all(|v| v.eq_approx(u)));
        }
    }

    #[test]
    fn write_ppm() {
        for &(width, height) in SIZES.iter() {
            let mut canvas = Canvas::new(width, height);

            for (u, &v) in canvas.iter_mut().zip(colors().iter().cycle()) {
                *u = v;
            }

            let mut buf = vec![];
            canvas.write_ppm(&mut buf).unwrap();

            let expected_data =
                colors()
                    .iter()
                    .cycle()
                    .take(width * height)
                    .fold(vec![], |mut data, u| {
                        data.extend_from_slice(&u.to_bytes());
                        data
                    });

            let lines = buf[..]
                .lines()
                .map(|line| line.unwrap())
                .collect::<Vec<String>>();

            assert!(
                lines[0] == "P3"
                    && lines[1] == format!("{} {}", width, height)
                    && lines[2] == "255"
            );

            let data = lines[3..lines.len() - 1]
                .iter()
                .flat_map(|line| line.split(' ').map(|num| num.parse::<u8>().unwrap()))
                .collect::<Vec<u8>>();

            assert!(
                data == expected_data
                    && lines[lines.len() - 1] == ""
                    && lines.iter().all(|line| line.len() <= 70)
            );
        }
    }
}
