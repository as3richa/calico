use crate::color::Color;
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

    pub fn set(&mut self, x: usize, y: usize, color: Color) {
        let i = self.index(x, y);
        self.data[i] = color;
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

    #[derive(Clone, Debug)]
    struct Size(usize);

    impl Size {
        const MAX_SIZE: usize = 256;
    }

    impl Arbitrary for Size {
        fn arbitrary(g: &mut Gen) -> Size {
            let size = <usize as Arbitrary>::arbitrary(g) % Size::MAX_SIZE + 1;
            Size(size)
        }
    }

    #[quickcheck]
    fn new(Size(width): Size, Size(height): Size) -> bool {
        let canvas = Canvas::new(width, height);
        let check = |x, y| canvas.get(x, y).eq_approx(Color::new(0.0, 0.0, 0.0));
        (0..width).all(|x| (0..height).all(|y| check(x, y)))
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
                canvas.iter().all(|color| color.eq_approx(u))
            }
    }
}
