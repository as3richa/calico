use crate::aabb::AABB;
use crate::bvh::Ray;
use crate::bvh::{Intersection, Primitive as BVHPrimitive, BVH};
use crate::color::Color;
use crate::matrix::Matrix;
use crate::shape::Shape;
use crate::tuple::{Tuple, Tuple3};
use crate::Canvas;
use crate::{consts, Float};

pub struct WorldBuilder {
    objects: Vec<PrimitiveBuilder>,
    prototypes: Vec<PrimitiveBuilder>,
    materials: Vec<Material>,
    lights: Vec<Light>,
    background: Color,
}

impl WorldBuilder {
    pub fn new() -> WorldBuilder {
        WorldBuilder {
            objects: vec![],
            prototypes: vec![],
            materials: vec![],
            lights: vec![],
            background: Color::new(0.0, 0.0, 0.0),
        }
    }

    pub fn object(&mut self, primitive: PrimitiveBuilder) {
        self.objects.push(primitive);
    }

    pub fn prototype(&mut self, primitive: PrimitiveBuilder) -> PrototypeHandle {
        let id = self.prototypes.len();
        self.prototypes.push(primitive);
        PrototypeHandle(id)
    }

    pub fn material(&mut self, material: Material) -> MaterialHandle {
        let id = self.materials.len();
        self.materials.push(material);
        MaterialHandle(id)
    }

    pub fn light(&mut self, light: Light) {
        self.lights.push(light);
    }

    pub fn set_background(&mut self, background: Color) {
        self.background = background;
    }

    pub fn finalize(self) -> World {
        let mut prototypes = Vec::with_capacity(self.prototypes.len());

        for primitive in self.prototypes {
            prototypes.push(primitive.finalize(&prototypes, &self.materials));
        }

        let mut objects = Vec::with_capacity(self.objects.len());

        for primitive in self.objects {
            objects.push(primitive.finalize(&prototypes, &self.materials))
        }

        World {
            bvh: BVH::new(&objects),
            lights: self.lights,
            background: self.background,
            _prototypes: prototypes,
            _materials: self.materials,
        }
    }
}

pub struct World {
    bvh: BVH<Primitive, SurfaceInteraction>,
    lights: Vec<Light>,
    background: Color,
    _prototypes: Vec<Primitive>,
    _materials: Vec<Material>,
}

impl World {
    pub fn cast_ray(&self, ray: Ray) -> Color {
        self.bvh
            .intersect_first(ray, Float::INFINITY)
            .map(|interaction| self.lighting(ray, interaction))
            .unwrap_or(self.background)
    }

    fn lighting(&self, ray: Ray, interaction: SurfaceInteraction) -> Color {
        let point = ray.at(interaction.time);
        let eye = (ray.origin - point).normalize();
        let normal = {
            let normal = interaction.normal.normalize();
            if normal.dot(eye) < 0.0 {
                -normal
            } else {
                normal
            }
        };

        let (color_m, ambient, diffuse_m, specular_m, shininess) = unsafe {
            let m = interaction.material;
            (
                (*m).color,
                (*m).ambient,
                (*m).diffuse,
                (*m).specular,
                (*m).shininess,
            )
        };

        let lighting_for_light = |light: &Light| {
            let Lighting {
                color: color_l,
                diffuse: diffuse_l,
                specular: specular_l,
            } = light.lighting(point, normal, eye);

            let diffuse_ambient_color =
                (color_l.blend(color_m)) * (ambient + diffuse_l * diffuse_m);

            let specular_color =
                color_l * Float::powf(Float::max(0.0, specular_l), shininess) * specular_m;

            diffuse_ambient_color + specular_color
        };

        self.lights
            .iter()
            .map(lighting_for_light)
            .fold(Color::new(0.0, 0.0, 0.0), |u, v| u + v)
    }
}

#[derive(Clone, Copy, Debug)]
pub struct PrototypeHandle(usize);

#[derive(Debug)]
pub struct PrimitiveBuilder {
    data: PrimitiveBuilderData,
    object_to_world: Matrix,
    casts_shadow: bool,
}

#[derive(Debug)]
enum PrimitiveBuilderData {
    Shape(Shape, MaterialHandle),
    Instance(PrototypeHandle),
}

impl PrimitiveBuilder {
    pub fn shape(shape: Shape, material: MaterialHandle) -> PrimitiveBuilder {
        PrimitiveBuilder {
            data: PrimitiveBuilderData::Shape(shape, material),
            object_to_world: Matrix::identity(),
            casts_shadow: true,
        }
    }

    pub fn instance(prototype: PrototypeHandle) -> PrimitiveBuilder {
        PrimitiveBuilder {
            data: PrimitiveBuilderData::Instance(prototype),
            object_to_world: Matrix::identity(),
            casts_shadow: true,
        }
    }

    pub fn translate(mut self, x: Float, y: Float, z: Float) -> PrimitiveBuilder {
        self.object_to_world.translate(x, y, z);
        self
    }

    pub fn rotate_x(mut self, x: Float) -> PrimitiveBuilder {
        self.object_to_world.rotate_x(x);
        self
    }

    pub fn rotate_y(mut self, y: Float) -> PrimitiveBuilder {
        self.object_to_world.rotate_y(y);
        self
    }

    pub fn rotate_z(mut self, z: Float) -> PrimitiveBuilder {
        self.object_to_world.rotate_z(z);
        self
    }

    pub fn scale(mut self, x: Float, y: Float, z: Float) -> PrimitiveBuilder {
        self.object_to_world.scale(x, y, z);
        self
    }

    pub fn shear(
        mut self,
        xy: Float,
        xz: Float,
        yx: Float,
        yz: Float,
        zx: Float,
        zy: Float,
    ) -> PrimitiveBuilder {
        self.object_to_world.shear(xy, xz, yx, yz, zx, zy);
        self
    }

    pub fn set_transform(mut self, object_to_world: Matrix) -> PrimitiveBuilder {
        self.object_to_world = object_to_world;
        self
    }

    pub fn set_casts_shadow(mut self, casts_shadow: bool) -> PrimitiveBuilder {
        self.casts_shadow = casts_shadow;
        self
    }

    fn finalize(self, prototypes: &[Primitive], materials: &[Material]) -> Primitive {
        let (data, aabb) = {
            let prototype_ptr = |id| {
                assert!(id < prototypes.len());
                unsafe { prototypes.as_ptr().add(id) }
            };
            let material_ptr = |id| {
                assert!(id < materials.len());
                unsafe { materials.as_ptr().add(id) }
            };

            match self.data {
                PrimitiveBuilderData::Shape(shape, MaterialHandle(id)) => {
                    (PrimitiveData::Shape(shape, material_ptr(id)), shape.aabb())
                }
                PrimitiveBuilderData::Instance(PrototypeHandle(id)) => (
                    PrimitiveData::Instance(prototype_ptr(id)),
                    prototypes[id].aabb,
                ),
            }
        };

        let mut transformed_aabb = AABB::empty();

        for i in 0..8 {
            let x = (if i & 1 == 0 { aabb.min } else { aabb.max })[0];
            let y = (if i & 2 == 0 { aabb.min } else { aabb.max })[1];
            let z = (if i & 4 == 0 { aabb.min } else { aabb.max })[2];
            let u = &self.object_to_world * Tuple::point(x, y, z);
            transformed_aabb = transformed_aabb.add(Tuple3::new([u.x, u.y, u.z]));
        }

        let world_to_object = self
            .object_to_world
            .inverse()
            .unwrap_or_else(Matrix::identity);

        let mut object_to_world_transpose = self.object_to_world;
        object_to_world_transpose.transpose();

        Primitive {
            data: data,
            aabb: transformed_aabb,
            object_to_world_transpose: object_to_world_transpose,
            world_to_object: world_to_object,
            casts_shadow: self.casts_shadow,
        }
    }
}

#[derive(Clone)]
struct Primitive {
    data: PrimitiveData,
    aabb: AABB,
    object_to_world_transpose: Matrix,
    world_to_object: Matrix,
    casts_shadow: bool,
}

#[derive(Clone)]
enum PrimitiveData {
    Shape(Shape, *const Material),
    Instance(*const Primitive),
}

impl BVHPrimitive<SurfaceInteraction> for Primitive {
    fn aabb(&self) -> AABB {
        self.aabb
    }

    fn intersect_first(&self, ray: Ray, max_time: Float) -> Option<SurfaceInteraction> {
        let transformed_ray = self.world_to_object.transform_ray(ray);

        let interaction = match self.data {
            PrimitiveData::Shape(shape, material) => shape
                .intersect_first(transformed_ray, max_time)
                .map(|intersection| SurfaceInteraction {
                    time: intersection.time,
                    normal: intersection.normal,
                    material: material,
                }),
            PrimitiveData::Instance(primitive) => unsafe {
                (*primitive).intersect_first(transformed_ray, max_time)
            },
        };

        interaction.map(|interaction| {
            let transformed_normal = self
                .world_to_object
                .mul_transpose_tuple(interaction.normal.as_vector())
                .as_tuple3();

            SurfaceInteraction {
                time: interaction.time,
                normal: transformed_normal,
                material: interaction.material,
            }
        })
    }
}

#[derive(Clone, Copy, Debug)]
pub struct MaterialHandle(usize);

pub struct Material {
    pub color: Color,
    pub diffuse: Float,
    pub ambient: Float,
    pub specular: Float,
    pub shininess: Float,
    pub reflectiveness: Float,
    pub transparency: Float,
    pub index_of_refraction: Float,
}

struct SurfaceInteraction {
    pub time: Float,
    pub normal: Tuple3,
    pub material: *const Material,
}

impl Intersection for SurfaceInteraction {
    fn time(&self) -> Float {
        self.time
    }
}

pub struct Light(LightData);

enum LightData {
    PointLight(Tuple3, Color),
    // spotlight, area light
}

struct Lighting {
    color: Color,
    diffuse: Float,
    specular: Float,
}

impl Light {
    pub fn point_light(position: Tuple3, intensity: Color) -> Light {
        Light(LightData::PointLight(position, intensity))
    }

    fn lighting(&self, point: Tuple3, normal: Tuple3, eye: Tuple3) -> Lighting {
        match self.0 {
            LightData::PointLight(position, color) => {
                let light = (position - point).normalize();
                let reflected = normal * light.dot(normal) * 2.0 - light;

                Lighting {
                    color: color,
                    diffuse: Float::max(light.dot(normal), 0.0),
                    specular: Float::max(reflected.dot(eye), 0.0),
                }
            }
        }
    }
}

pub struct Camera {
    field_of_view: Float,
    focal_distance: Float,
    world_to_camera: Matrix,
    // aperture
}

impl Camera {
    pub fn new() -> Camera {
        Camera {
            field_of_view: consts::FRAC_PI_2,
            focal_distance: 1.0,
            world_to_camera: Matrix::identity(),
        }
    }

    pub fn set_field_of_view(mut self, field_of_view: Float) -> Camera {
        self.field_of_view = field_of_view;
        self
    }

    pub fn set_focal_distance(mut self, focal_distance: Float) -> Camera {
        self.focal_distance = focal_distance;
        self
    }

    pub fn look_at(mut self, eye: Tuple3, center: Tuple3, up: Tuple3) -> Camera {
        self.world_to_camera = Matrix::look_at(eye, center, up);
        self
    }

    pub fn set_transform(mut self, world_to_camera: Matrix) -> Camera {
        self.world_to_camera = world_to_camera;
        self
    }

    pub fn render(&self, world: &World, width: usize, height: usize) -> Canvas {
        let camera_to_world = self
            .world_to_camera
            .inverse()
            .unwrap_or_else(Matrix::identity);

        let pixel_size = {
            let len = usize::max(width, height) as Float;
            2.0 * self.focal_distance * Float::tan(self.field_of_view / 2.0) / len
        };

        let left = -(width as Float) / 2.0 * pixel_size;
        let top = (height as Float) / 2.0 * pixel_size;

        let mut canvas = Canvas::new(width, height);

        for j in 0..height {
            for i in 0..width {
                let x = left + pixel_size * (i as Float + 0.5);
                let y = top - pixel_size * (j as Float + 0.5);

                let camera_ray = Ray::new(
                    Tuple3::new([0.0, 0.0, 0.0]),
                    Tuple3::new([x, y, self.focal_distance]),
                );

                let ray = camera_to_world.transform_ray(camera_ray);
                canvas.set(i, j, world.cast_ray(ray));
            }
        }

        canvas
    }
}
