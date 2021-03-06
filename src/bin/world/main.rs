extern crate calico;

use calico::shape::Shape;
use calico::tuple::Tuple3;
use calico::world::{Camera, Light, Material, PrimitiveBuilder, WorldBuilder};
use calico::Color;

fn main() {
    let mut builder = WorldBuilder::new();

    let m_wall = builder.material(Material {
        color: Color::new(1.0, 0.9, 0.9),
        diffuse: 0.9,
        ambient: 0.1,
        specular: 0.0,
        shininess: 200.0,
        reflectiveness: 0.0,
        transparency: 0.0,
        refractive_index: None,
        priority: 0,
        casts_shadow: false,
    });

    let m_mirror = builder.material(Material {
        color: Color::new(0.0, 0.0, 0.0),
        diffuse: 0.0,
        ambient: 0.0,
        specular: 0.1,
        shininess: 200.0,
        reflectiveness: 0.75,
        transparency: 0.0,
        refractive_index: None,
        priority: 0,
        casts_shadow: false,
    });

    let m_left_sphere = builder.material(Material {
        color: Color::new(1.0, 0.8, 0.1),
        diffuse: 0.7,
        ambient: 0.1,
        specular: 0.3,
        shininess: 200.0,
        reflectiveness: 0.0,
        transparency: 0.0,
        refractive_index: None,
        priority: 0,
        casts_shadow: false,
    });

    let m_middle_sphere = builder.material(Material {
        color: Color::new(1.0, 0.0, 1.0),
        diffuse: 0.1,
        ambient: 0.0,
        specular: 0.0,
        shininess: 200.0,
        reflectiveness: 0.01,
        transparency: 0.9,
        refractive_index: Some(1.52),
        priority: 0,
        casts_shadow: false,
    });

    let m_right_sphere = builder.material(Material {
        color: Color::new(0.5, 1.0, 0.1),
        diffuse: 0.5,
        ambient: 0.1,
        specular: 0.3,
        shininess: 200.0,
        reflectiveness: 0.0,
        transparency: 0.0,
        refractive_index: None,
        priority: 0,
        casts_shadow: false,
    });

    let m_air = builder.material(Material {
        color: Color::new(0.5, 1.0, 0.1),
        diffuse: 0.0,
        ambient: 0.0,
        specular: 0.0,
        shininess: 0.0,
        reflectiveness: 0.0,
        transparency: 1.0,
        refractive_index: Some(1.0),
        priority: 1,
        casts_shadow: false,
    });

    builder.object(PrimitiveBuilder::shape(
        Shape::triangle(
            Tuple3::new2(0.0, 0.0, 0.0),
            Tuple3::new2(1.0, 0.0, 0.0),
            Tuple3::new2(0.0, 1.0, 0.0),
        ),
        m_wall,
    ));
    builder.object(PrimitiveBuilder::shape(
        Shape::triangle(
            Tuple3::new2(1.0, 0.0, 0.0),
            Tuple3::new2(0.0, 1.0, 0.0),
            Tuple3::new2(1.0, 1.0, 0.0),
        ),
        m_wall,
    ));
    builder.object(PrimitiveBuilder::shape(
        Shape::triangle(
            Tuple3::new2(0.0, 0.0, 0.0),
            Tuple3::new2(0.0, 0.0, 1.0),
            Tuple3::new2(0.0, 1.0, 0.0),
        ),
        m_wall,
    ));
    builder.object(PrimitiveBuilder::shape(
        Shape::triangle(
            Tuple3::new2(0.0, 0.0, 1.0),
            Tuple3::new2(0.0, 1.0, 0.0),
            Tuple3::new2(0.0, 1.0, 1.0),
        ),
        m_wall,
    ));
    builder.object(PrimitiveBuilder::shape(
        Shape::triangle(
            Tuple3::new2(0.0, 0.0, 0.0),
            Tuple3::new2(0.0, 0.0, 1.0),
            Tuple3::new2(1.0, 0.0, 0.0),
        ),
        m_mirror,
    ));
    builder.object(PrimitiveBuilder::shape(
        Shape::triangle(
            Tuple3::new2(0.0, 0.0, 1.0),
            Tuple3::new2(1.0, 0.0, 0.0),
            Tuple3::new2(1.0, 0.0, 1.0),
        ),
        m_mirror,
    ));
    builder.object(PrimitiveBuilder::shape(
        Shape::triangle(
            Tuple3::new2(0.0, 1.0, 0.0),
            Tuple3::new2(0.0, 1.0, 1.0),
            Tuple3::new2(1.0, 1.0, 0.0),
        ),
        m_wall,
    ));
    builder.object(PrimitiveBuilder::shape(
        Shape::triangle(
            Tuple3::new2(0.0, 1.0, 1.0),
            Tuple3::new2(1.0, 1.0, 0.0),
            Tuple3::new2(1.0, 1.0, 1.0),
        ),
        m_wall,
    ));

    builder.object(
        PrimitiveBuilder::shape(Shape::sphere(0.08), m_left_sphere).translate(0.8, 0.2, 0.2),
    );

    builder.object(
        PrimitiveBuilder::shape(Shape::sphere(0.15), m_middle_sphere).translate(0.5, 0.5, 0.5),
    );
    builder.object(PrimitiveBuilder::shape(Shape::sphere(0.07), m_air).translate(0.5, 0.5, 0.5));

    builder.object(
        PrimitiveBuilder::shape(Shape::sphere(1.0), m_right_sphere).translate(0.7, 0.7, 1.0),
    );

    builder.light(Light::point_light(
        Tuple3::new([0.5, 0.01, 0.5]),
        Color::new(1.0, 1.0, 1.0),
    ));

    let world = builder.finalize();

    let camera = Camera::new().set_field_of_view(3.14159 / 3.0).look_at(
        Tuple3::new([0.3, 0.3, 0.02]),
        Tuple3::new([0.7, 0.7, 1.0]),
        Tuple3::new([0.0, 1.0, 0.0]),
    );

    let height = 350;
    let canvas = camera.render(&world, height, height);

    let mut f = std::fs::OpenOptions::new()
        .write(true)
        .create(true)
        .open("spheres.ppm")
        .unwrap();

    canvas.write_ppm(&mut f).unwrap();
}
