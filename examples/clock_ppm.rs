extern crate calico;

use calico::{consts, Canvas, Color, Float, Matrix, Tuple};
use std::fs;

fn main() {
    let mut canvas = Canvas::new(500, 500);
    let color = Color::new(1.0, 0.0, 1.0);

    for i in 0..12 {
        let angle = 2.0 * consts::PI * (i as Float) / 12.0;
        let mut m = Matrix::rotation_z(angle);
        m.scale(220.0, 220.0, 0.0);
        m.translate(250.0, 250.0, 0.0);
        let u = &m * Tuple::point(0.0, 1.0, 0.0);

        canvas.set(u.x.round() as usize, u.y.round() as usize, color);
    }

    let mut file = fs::OpenOptions::new()
        .create(true)
        .write(true)
        .open("clock.ppm")
        .unwrap();

    canvas.write_ppm(&mut file).unwrap();
}
