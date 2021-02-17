use crate::aabb::AABB;
use crate::bvh::Ray;
use crate::bvh::{Intersection, Primitive as BVHPrimitive, BVH};
use crate::color::Color;
use crate::matrix::Matrix;
use crate::shape::Shape;
use crate::tuple::{Tuple, Tuple3};
use crate::Float;

pub struct WorldBuilder {
    objects: Vec<PrimitiveBuilder>,
    prototypes: Vec<PrimitiveBuilder>,
    materials: Vec<Material>,
}

impl WorldBuilder {
    pub fn new() -> WorldBuilder {
        WorldBuilder {
            objects: vec![],
            prototypes: vec![],
            materials: vec![],
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
            prototypes: prototypes,
            materials: self.materials,
        }
    }
}

pub struct World {
    bvh: BVH<Primitive, SurfaceInteraction>,
    prototypes: Vec<Primitive>,
    materials: Vec<Material>,
}

impl World {
    pub fn cast_ray(&self, ray: Ray, max_time: Float) -> Option<SurfaceInteraction> {
        self.bvh.intersect_first(ray, max_time)
    }

    pub fn cast_shadow_ray(&self, ray: Ray, max_time: Float) -> bool {
        unimplemented!()
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
    Transformed(PrototypeHandle),
}

impl PrimitiveBuilder {
    pub fn shape(shape: Shape, material: MaterialHandle) -> PrimitiveBuilder {
        PrimitiveBuilder {
            data: PrimitiveBuilderData::Shape(shape, material),
            object_to_world: Matrix::identity(),
            casts_shadow: true,
        }
    }

    pub fn transformed(prototype: PrototypeHandle) -> PrimitiveBuilder {
        PrimitiveBuilder {
            data: PrimitiveBuilderData::Transformed(prototype),
            object_to_world: Matrix::identity(),
            casts_shadow: true,
        }
    }

    pub fn translate(&mut self, x: Float, y: Float, z: Float) {
        self.object_to_world.translate(x, y, z);
    }

    pub fn rotate_x(&mut self, x: Float) {
        self.object_to_world.rotate_x(x);
    }

    pub fn rotate_y(&mut self, y: Float) {
        self.object_to_world.rotate_y(y);
    }

    pub fn rotate_z(&mut self, z: Float) {
        self.object_to_world.rotate_z(z);
    }

    pub fn scale(&mut self, x: Float, y: Float, z: Float) {
        self.object_to_world.scale(x, y, z);
    }

    pub fn shear(&mut self, xy: Float, xz: Float, yx: Float, yz: Float, zx: Float, zy: Float) {
        self.object_to_world.shear(xy, xz, yx, yz, zx, zy);
    }

    pub fn set_casts_shadow(&mut self, casts_shadow: bool) {
        self.casts_shadow = casts_shadow;
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
                PrimitiveBuilderData::Transformed(PrototypeHandle(id)) => (
                    PrimitiveData::Transformed(prototype_ptr(id)),
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
    Transformed(*const Primitive),
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
            PrimitiveData::Transformed(primitive) => unsafe {
                (*primitive).intersect_first(transformed_ray, max_time)
            },
        };

        interaction.map(|interaction| {
            let mut foo = self.world_to_object.clone();
            foo.transpose();
            let transformed_normal = &foo * interaction.normal.as_vector();
            SurfaceInteraction {
                time: interaction.time,
                normal: transformed_normal.as_tuple3(),
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

pub struct SurfaceInteraction {
    pub time: Float,
    pub normal: Tuple3,
    pub material: *const Material,
}

impl Intersection for SurfaceInteraction {
    fn time(&self) -> Float {
        self.time
    }
}
