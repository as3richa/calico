extern crate calico;

use calico::bvh::Ray;
use calico::scene::{Light, Lighting};
use calico::shape::Shape;
use calico::tuple::Tuple3;
use calico::world::{Material, PrimitiveBuilder, SurfaceInteraction, WorldBuilder};
use calico::Canvas;
use calico::Color;
use calico::Float;

fn main() {
    let mut builder = WorldBuilder::new();

    let material = builder.material(Material {
        color: Color::new(1.0, 0.0, 1.0),
        diffuse: 0.9,
        ambient: 0.1,
        specular: 0.9,
        shininess: 200.0,
        reflectiveness: 0.0,
        transparency: 0.0,
        index_of_refraction: 1.0,
    });

    let sphere = {
        let mut sphere = PrimitiveBuilder::shape(Shape::Sphere, material);
        sphere.scale(6.0, 1.0, 6.0);
        builder.prototype(sphere)
    };

    for x in 0..1 {
        for y in 0..1 {
            let mut transformed = PrimitiveBuilder::transformed(sphere);
            transformed.translate(x as Float, y as Float, 0.0);
            transformed.rotate_x(3.14159 *1.50);
            builder.object(transformed);
        }
    }

    let world = builder.finalize();

    let light = Light::PointLight(
        Tuple3::new([-  100.0, 100.0, -100.0]),
        Color::new(1.0, 1.0, 1.0),
    );

    let size = 250;
    let mut canvas = Canvas::new(size, size);

    for i in 0..size {
        for j in 0..size {
            let x = -7.0 + (i as Float) / (size as Float) * 14.0;
            let y = 7.0 - (j as Float) / (size as Float) * 14.0;

            let camera = Tuple3::new([0.0, 0.0, -20.0]);
            let ray = Ray::new(camera, Tuple3::new([x, y, 20.0]));

            if let Some(interaction) = world.cast_ray(ray, Float::INFINITY) {
                let SurfaceInteraction {
                    time,
                    normal: unnormalized,
                    material,
                } = interaction;

                let normal = unnormalized.normalize();
                let point = ray.at(time);
                let eye = (camera - point).normalize();

                let color = unsafe {
                    let Lighting {
                        color: light_color,
                        diffuse,
                        specular,
                    } = light.lighting(point, normal, eye, (*material).shininess);

                    let ambient_diffuse_component = (*material).color.blend(light_color)
                        * ((*material).ambient + diffuse * (*material).diffuse);

                    let specular_component = if (*material).specular.abs() > 1e-4 {
                        // FIXME: epsilon??
                        light_color * (specular * (*material).specular)
                    } else {
                        Color::new(0.0, 0.0, 0.0)
                    };

                    ambient_diffuse_component + specular_component
                };

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
