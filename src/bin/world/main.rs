extern crate calico;

use calico::bvh::Ray;
use calico::shape::Shape;
use calico::tuple::Tuple3;
use calico::world::{Material, PrimitiveBuilder, SurfaceInteraction, WorldBuilder};
use calico::Canvas;
use calico::Color;
use calico::Float;

fn main() {
    let mut builder = WorldBuilder::new();

    let material = builder.material(Material {
        color: Color::new(1.0, 0.0, 0.0),
        diffuse: 0.7,
        ambient: 0.1,
        specular: 0.0,
        shininess: 80.0,
        reflectiveness: 0.0,
        transparency: 0.0,
        index_of_refraction: 1.0,
    });

    let sphere = {
        let mut sphere = PrimitiveBuilder::shape(Shape::Sphere, material);
        sphere.scale(0.25, 0.25, 0.25);
        builder.prototype(sphere)
    };

    for x in 0..1000 {
        for y in 0..1000 {
            let mut transformed = PrimitiveBuilder::transformed(sphere);
            transformed.translate(x as Float, y as Float, (x + y) as Float % 2.0);
            builder.object(transformed);
        }
    }

    let world = builder.finalize();
    let mut canvas = Canvas::new(1000, 1000);

    for i in 0..1000 {
        for j in 0..1000 {
            let x = -5.0 + (i as Float) / 50.0;
            let y = 15.0 - (j as Float) / 50.0;

            if let Some(interaction) = world.cast_ray(
                Ray::new(Tuple3::new([x, y, 20.0]), Tuple3::new([0.0, 0.0, -1.0])),
                Float::INFINITY,
            ) {
                let color = unsafe { (*interaction.material).color };
                canvas.set(i, j, color);
            }
        }
    }

    let mut f = std::fs::OpenOptions::new()
        .write(true)
        .create(true)
        .open("spheres.ppm")
        .unwrap();
    canvas.write_ppm(&mut f).unwrap();
}
