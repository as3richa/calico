extern crate calico;

use calico::shape::Shape;
use calico::tuple::Tuple3;
use calico::world::{Camera, Light, Material, PrimitiveBuilder, WorldBuilder};
use calico::Color;
use calico::Matrix;

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

    let mut floor = PrimitiveBuilder::shape(Shape::Sphere, m_floor);
    floor.scale(10.0, 0.01, 10.0);
    builder.object(floor);

    let mut left_wall = PrimitiveBuilder::shape(Shape::Sphere, m_floor);
    left_wall.scale(10.0, 0.01, 10.0);
    left_wall.rotate_x(3.1415 / 2.0);
    left_wall.rotate_y(-3.1415 / 4.0);
    left_wall.translate(0.0, 0.0, 5.0);
    builder.object(left_wall);

    let mut right_wall = PrimitiveBuilder::shape(Shape::Sphere, m_floor);
    right_wall.scale(10.0, 0.01, 10.0);
    right_wall.rotate_x(3.1415 / 2.0);
    right_wall.rotate_y(3.1415 / 4.0);
    right_wall.translate(0.0, 0.0, 5.0);
    builder.object(right_wall);

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

    let mut middle = PrimitiveBuilder::shape(Shape::Sphere, m_middle);
    middle.translate(-0.5, 1.0, 0.5);
    builder.object(middle);

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

    let mut right = PrimitiveBuilder::shape(Shape::Sphere, m_right);
    right.scale(0.5, 0.5, 0.5);
    right.translate(1.5, 0.5, -0.5);
    builder.object(right);

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

    let mut left = PrimitiveBuilder::shape(Shape::Sphere, m_left);
    left.scale(1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0);
    left.translate(-1.5, 1.0 / 3.0, -0.75);
    builder.object(left);

    builder.light(Light::PointLight(
        Tuple3::new([-10.0, 10.0, -10.0]),
        Color::new(1.0, 1.0, 1.0),
    ));

    let world = builder.finalize();

    let camera = Camera::new(
        1000,
        500,
        3.1415 / 3.0,
        1.0,
        Matrix::look_at(
            Tuple3::new([0.0, 1.5, -5.0]),
            Tuple3::new([0.0, 1.0, 0.0]),
            Tuple3::new([0.0, 1.0, 0.0]),
        ),
    );

    let canvas = camera.render(&world);

    let mut f = std::fs::OpenOptions::new()
        .write(true)
        .create(true)
        .open("spheres.ppm")
        .unwrap();

    canvas.write_ppm(&mut f).unwrap();
}
