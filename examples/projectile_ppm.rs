extern crate calico;

use calico::{Canvas, Color, Float, Tuple};
use std::fs;

fn main() {
    let initial_pos = Tuple::point(0.0, 1.0, 0.0);
    let initial_vel = Tuple::vector(5.5, 10.0, 0.0);
    let gravity = Tuple::vector(0.0, -0.1, 0.0);
    let wind = Tuple::vector(-0.01, 0.0, 0.0);

    let mut canvas = Canvas::new(900, 550);
    let color = Color::new(1.0, 0.0, 1.0);

    let mut pos = initial_pos;
    let mut vel = initial_vel;

    while pos.y > 0.0 {
        let canvas_x = pos.x.round();
        let canvas_y = (canvas.height() - 1) as Float - pos.y.round();

        if 0.0 <= canvas_x
            && canvas_x < canvas.width() as Float
            && 0.0 <= canvas_y
            && canvas_y < canvas.height() as Float
        {
            canvas.set(canvas_x as usize, canvas_y as usize, color);
        }

        pos += vel;
        vel += gravity + wind;
    }

    let mut file = fs::OpenOptions::new()
        .create(true)
        .write(true)
        .open("projectile.ppm")
        .unwrap();

    canvas.write_ppm(&mut file).unwrap();
}
