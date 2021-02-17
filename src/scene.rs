use crate::canvas::Canvas;
use crate::color::Color;
use crate::tuple::Tuple3;
use crate::world::{Material, World};
use crate::Float;

pub struct Camera {}

#[derive(Clone, Copy)]
pub enum Light {
    PointLight(Tuple3, Color),
    // spotlight, area light
}

pub struct Lighting {
    pub color: Color,
    pub diffuse: Float,
    pub specular: Float,
}

impl Light {
    pub fn lighting(
        self,
        point: Tuple3,
        normal: Tuple3,
        eye: Tuple3,
        shininess: Float,
    ) -> Lighting {
        match self {
            Light::PointLight(position, color) => {
                let light = (position - point).normalize();
                let reflected = normal * light.dot(normal) * 2.0 - light;

                Lighting {
                    color: color,
                    diffuse: Float::max(light.dot(normal), 0.0),
                    specular: Float::powf(Float::max(reflected.dot(eye), 0.0), shininess),
                }
            }
        }
    }
}

pub struct Scene {
    lights: Vec<Light>,
    world: World,
    background: Color,
    camera: Camera,
}

impl Scene {
    fn new(lights: Vec<Light>, world: World, background: Color, camera: Camera) -> Scene {
        Scene {
            lights: lights,
            world: world,
            background: background,
            camera: camera,
        }
    }

    fn render(canvas: &Canvas) {}
}
