use crate::aabb::AABB;
use crate::arena::Arena;
use crate::bvh::{Intersection as BVHIntersection, Primitive as BVHPrimitive, Ray};
use crate::color::Color;
use crate::matrix::Matrix;
use crate::tuple::{Tuple, Tuple3};
use crate::Float;

#[derive(Clone, Copy)]
enum Shape {
    Sphere,
    // Cone,
    // Cube,
    // Cylinder(bool, bool),
    // InfiniteCylinder,
    // Plane,
    // SemiInfiniteCylinder(bool),
    // Torus,
    // Triangle((Tuple3, Tuple3, Tuple3), (Tuple2, Tuple2, Tuple2)),
}

struct ShapeIntersection {
    time: Float,
    point: Tuple3,
    normal: Tuple3,
}

impl Shape {
    fn aabb(self) -> AABB {
        match self {
            Shape::Sphere => AABB::new(
                Tuple3::new([-1.0, -1.0, -1.0]),
                Tuple3::new([1.0, 1.0, 1.0]),
            ),
        }
    }

    fn intersect_first(self, ray: Ray, max_time: Float) -> Option<ShapeIntersection> {
        match self {
            Shape::Sphere => Shape::intersect_unit_sphere(ray, max_time),
        }
    }

    fn intersect_unit_sphere(ray: Ray, max_time: Float) -> Option<ShapeIntersection> {
        let oo = ray.origin.dot(ray.origin);
        let ov = ray.origin.dot(ray.velocity);
        let vv = ray.velocity.dot(ray.velocity);
        let radicand = ov * ov - vv * (oo - 1.0);

        let time = if radicand < 0.0 {
            None
        } else {
            let radical = Float::sqrt(radicand);
            let minus = (-ov - radical) / vv;

            if minus < max_time {
                if minus >= 0.0 {
                    Some(minus)
                } else {
                    let plus = (-ov + radical) / vv;
                    if 0.0 <= plus && plus < max_time {
                        Some(plus)
                    } else {
                        None
                    }
                }
            } else {
                None
            }
        };

        time.map(|time| {
            let point = ray.at(time);

            ShapeIntersection {
                time: time,
                point: point,
                normal: point.normalize(),
            }
        })
    }
}

enum PrimitiveData<'a> {
    Shape(Shape, &'a Diffuse<'a>),
    // AnimatedInstance(&'a Self, Matrix),
    // Difference(&'a Self, &'a Self),
    // Instance(&'a Self),
    // Intersection(&'a Self, &'a Self),
    // Union(&'a Self, &'a Self),
}

struct Primitive<'a> {
    data: PrimitiveData<'a>,
    world_to_object: Matrix,
    object_to_world: Matrix,
}

impl<'a> Primitive<'a> {
    pub fn translate(&mut self, x: Float, y: Float, z: Float) {
        self.world_to_object.translate(x, y, z);
    }

    pub fn rotate_x(&mut self, x: Float) {
        self.world_to_object.rotate_x(x);
    }

    pub fn rotate_y(&mut self, y: Float) {
        self.world_to_object.rotate_y(y);
    }

    pub fn rotate_z(&mut self, z: Float) {
        self.world_to_object.rotate_x(z);
    }

    pub fn scale(&mut self, x: Float, y: Float, z: Float) {
        self.world_to_object.scale(x, y, z);
    }

    pub fn shear(&mut self, xy: Float, xz: Float, yx: Float, yz: Float, zx: Float, zy: Float) {
        self.world_to_object.shear(xy, xz, yx, yz, zx, zy);
    }

    fn compute_object_to_world(&mut self) {
        self.object_to_world = self.world_to_object.clone();
        self.object_to_world.invert();
    }

    fn transform_aabb(&self, aabb: AABB) -> AABB {
        let mut transformed = AABB::empty();

        for i in 0..8 {
            let x = (if i & 1 == 0 { aabb.min } else { aabb.max })[0];
            let y = (if i & 2 == 0 { aabb.min } else { aabb.max })[1];
            let z = (if i & 4 == 0 { aabb.min } else { aabb.max })[2];

            let u = &self.world_to_object * Tuple::point(x, y, z);
            transformed = transformed.add(Tuple3::new([u.x, u.y, u.z]));
        }

        transformed
    }
}

struct Intersection {
    time: Float,
    point: Tuple3,
    normal: Tuple3,
    diffuse: Color,
}

impl Intersection {
    fn new(time: Float, point: Tuple3, normal: Tuple3, diffuse: Color) -> Intersection {
        Intersection {
            time: time,
            point: point,
            normal: normal,
            diffuse: diffuse,
        }
    }
}

impl<'a> BVHIntersection for Intersection {
    fn time(&self) -> Float {
        self.time
    }
}

impl<'a> BVHPrimitive<Intersection> for &'a Primitive<'a> {
    fn aabb(self) -> AABB {
        match self.data {
            PrimitiveData::Shape(shape, _) => self.transform_aabb(shape.aabb()),
        }
    }

    fn intersect_first(self, ray: Ray, max_time: Float) -> Option<Intersection> {
        match self.data {
            PrimitiveData::Shape(shape, diffuse) => {
                shape.intersect_first(ray, max_time).map(|intersection| {
                    let diffuse = match diffuse {
                        Diffuse::Pattern(pattern) => pattern.sample(intersection.point),
                    };
                    Intersection::new(
                        intersection.time,
                        intersection.point,
                        intersection.normal,
                        diffuse,
                    )
                })
            }
        }
    }
}

/*
enum Texture<'a> {
    Function(&'a dyn Fn(Tuple2) -> Color),
    Canvas(Canvas),
}
*/

enum Pattern<'a> {
    Function(&'a dyn Fn(Tuple3) -> Color),
    // Gradient(Color, Color),
    // Rings(Color, Color),
    Solid(Color),
    // Stripes(Color),
    // Transformed(&'a Self, Matrix),
}

impl<'a> Pattern<'a> {
    fn sample(&self, point: Tuple3) -> Color {
        match self {
            Pattern::Function(f) => f(point),
            Pattern::Solid(u) => *u,
        }
    }
}

enum Diffuse<'a> {
    // Texture(Texture<'a>),
    Pattern(Pattern<'a>),
}

struct WorldBuilder<'a> {
    primitives: Arena<Primitive<'a>>,
    diffuses: Arena<Diffuse<'a>>,
    to_render: Vec<&'a Primitive<'a>>,
}

impl<'a> WorldBuilder<'a> {
    fn shape(&self, shape: Shape, diffuse: &'a Diffuse<'a>) -> &Primitive {
        self.primitives.alloc(Primitive {
            data: PrimitiveData::Shape(shape, diffuse),
            world_to_object: Matrix::identity(),
            object_to_world: Matrix::identity(),
        })
    }

    fn solid(&self, u: Color) -> &Diffuse<'a> {
        self.diffuses.alloc(Diffuse::Pattern(Pattern::Solid(u)))
    }

    fn add(&mut self, primitive: &'a Primitive) {
        self.to_render.push(primitive);
    }
}
