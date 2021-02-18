extern crate calico;

use calico::shape::Shape;
use calico::tuple::Tuple3;
use calico::world::{Camera, Light, Material, PrimitiveBuilder, WorldBuilder};
use calico::Color;

fn main() {
    let mut builder = WorldBuilder::new();

    let m_floor = builder.material(Material {
        color: Color::new(1.0, 0.9, 0.9),
        diffuse: 0.9,
        ambient: 0.1,
        specular: 0.0,
        shininess: 200.0,
        reflectiveness: 0.0,
        transparency: 0.0,
        index_of_refraction: 1.0,
    });

    builder.object(PrimitiveBuilder::shape(Shape::Sphere, m_floor).scale(10.0, 0.01, 10.0));

    builder.object(
        PrimitiveBuilder::shape(Shape::Sphere, m_floor)
            .scale(10.0, 0.01, 10.0)
            .rotate_x(3.1415 / 2.0)
            .rotate_y(-3.1415 / 4.0)
            .translate(0.0, 0.0, 5.0),
    );

    builder.object(
        PrimitiveBuilder::shape(Shape::Sphere, m_floor)
            .scale(10.0, 0.01, 10.0)
            .rotate_x(3.1415 / 2.0)
            .rotate_y(3.1415 / 4.0)
            .translate(0.0, 0.0, 5.0),
    );

    let m_middle = builder.material(Material {
        color: Color::new(0.1, 1.0, 0.5),
        diffuse: 0.7,
        ambient: 0.1,
        specular: 0.3,
        shininess: 200.0,
        reflectiveness: 0.0,
        transparency: 0.0,
        index_of_refraction: 1.0,
    });

    builder.object(PrimitiveBuilder::shape(Shape::Sphere, m_middle).translate(-0.5, 1.0, 0.5));

    let m_right = builder.material(Material {
        color: Color::new(0.5, 1.0, 0.1),
        diffuse: 0.7,
        ambient: 0.1,
        specular: 0.3,
        shininess: 200.0,
        reflectiveness: 0.0,
        transparency: 0.0,
        index_of_refraction: 1.0,
    });

    builder.object(
        PrimitiveBuilder::shape(Shape::Sphere, m_right)
            .scale(0.5, 0.5, 0.5)
            .translate(1.5, 0.5, -0.5),
    );

    let m_left = builder.material(Material {
        color: Color::new(1.0, 0.8, 0.1),
        diffuse: 0.7,
        ambient: 0.1,
        specular: 0.3,
        shininess: 200.0,
        reflectiveness: 0.0,
        transparency: 0.0,
        index_of_refraction: 1.0,
    });

    builder.object(
        PrimitiveBuilder::shape(Shape::Sphere, m_left)
            .scale(1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0)
            .translate(-1.5, 1.0 / 3.0, -0.75),
    );

    builder.light(Light::PointLight(
        Tuple3::new([-10.0, 10.0, -10.0]),
        Color::new(1.0, 1.0, 1.0),
    ));

    builder.light(Light::PointLight(
        Tuple3::new([10.0, 10.0, -10.0]),
        Color::new(1.0, 1.0,1.0),
    ));

    let world = builder.finalize();

    let camera = Camera::new().set_field_of_view(3.14159 / 3.0).look_at(
        Tuple3::new([0.0, 1.5, -5.0]),
        Tuple3::new([0.0, 1.0, 0.0]),
        Tuple3::new([0.0, 1.0, 0.0]),
    );

    let canvas = camera.render(&world, 1000, 500);

    let mut f = std::fs::OpenOptions::new()
        .write(true)
        .create(true)
        .open("spheres.ppm")
        .unwrap();

    canvas.write_ppm(&mut f).unwrap();
}
