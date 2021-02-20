use crate::aabb::AABB;
use crate::bvh::Ray;
use crate::tuple::Tuple3;
use crate::Float;

#[derive(Clone, Copy, Debug)]
pub enum Shape {
    Sphere(Float),
    Triangle([Tuple3; 3]),
    SmoothTriangle([Tuple3; 3], [Tuple3; 3]),
    // Cone,
    // Cube,
    // Cylinder(bool, bool),
    // InfiniteCylinder,
    // Plane,
    // SemiInfiniteCylinder(bool),
    // Torus
}

pub struct ShapeIntersection {
    pub time: Float,
    pub normal: Tuple3,
}

pub enum ShapeBounds {
    AABB(AABB),
    Triangle([Tuple3; 3]),
}

impl Shape {
    pub fn sphere(radius: Float) -> Shape {
        Shape::Sphere(radius)
    }

    pub fn triangle(p: Tuple3, q: Tuple3, r: Tuple3) -> Shape {
        Shape::Triangle([p, q, r])
    }

    pub fn bounds(self) -> ShapeBounds {
        match self {
            Shape::Sphere(radius) => ShapeBounds::AABB(AABB::new(
                Tuple3::new([-radius, -radius, -radius]),
                Tuple3::new([radius, radius, radius]),
            )),
            Shape::Triangle(vertices) => ShapeBounds::Triangle(vertices),
            Shape::SmoothTriangle(vertices, _) => ShapeBounds::Triangle(vertices),
        }
    }

    pub fn intersect_first(self, ray: Ray, max_time: Float) -> Option<ShapeIntersection> {
        match self {
            Shape::Sphere(radius) => sphere::intersect_first(radius, ray, max_time).map(|time| {
                let point = ray.at(time);
                ShapeIntersection {
                    time: time,
                    normal: sphere::normal(point),
                }
            }),
            Shape::Triangle(vertices) => {
                // fixme: vertices ref?? idgi
                triangle::intersect_first(vertices, ray, max_time).map(|intersection| {
                    // FIXME: precalculate normal
                    let normal = (vertices[1] - vertices[0]).cross(vertices[2] - vertices[0]);
                    ShapeIntersection {
                        time: intersection.time,
                        normal: normal,
                    }
                })
            }
            Shape::SmoothTriangle(vertices, normals) => {
                triangle::intersect_first(vertices, ray, max_time).map(|intersection| {
                    // FIXME: precalculate normal differences?
                    let normal = normals[0]
                        + (normals[1] - normals[0]) * intersection.u
                        + (normals[2] - normals[0]) * intersection.v;
                    ShapeIntersection {
                        time: intersection.time,
                        normal: normal,
                    }
                })
            }
        }
    }

    pub fn intersect_pred(self, ray: Ray, max_time: Float) -> bool {
        match self {
            Shape::Sphere(radius) => sphere::intersect_first(radius, ray, max_time).is_some(),
            Shape::Triangle(vertices) => {
                triangle::intersect_first(vertices, ray, max_time).is_some()
            }
            Shape::SmoothTriangle(vertices, _) => {
                triangle::intersect_first(vertices, ray, max_time).is_some()
            }
        }
    }
}

mod sphere {
    use crate::bvh::Ray;
    use crate::tuple::Tuple3;
    use crate::Float;

    pub fn intersect_first(radius: Float, ray: Ray, max_time: Float) -> Option<Float> {
        let oo = ray.origin.dot(ray.origin);
        let ov = ray.origin.dot(ray.velocity);
        let vv = ray.velocity.dot(ray.velocity);
        let radicand = ov * ov - vv * (oo - radius * radius);

        if radicand < 0.0 {
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
        }
    }

    pub fn normal(point: Tuple3) -> Tuple3 {
        point
    }
}

mod triangle {
    use crate::bvh::Ray;
    use crate::tuple::Tuple3;
    use crate::Float;

    pub struct TriangleIntersection {
        pub time: Float,
        pub u: Float,
        pub v: Float,
    }

    pub fn intersect_first(
        vertices: [Tuple3; 3],
        ray: Ray,
        max_time: Float,
    ) -> Option<TriangleIntersection> {
        let epsilon = 1e-5; // FIXME: too many ad hoc epsilons

        let ba = vertices[1] - vertices[0];
        let ca = vertices[2] - vertices[0];

        let p = ray.velocity.cross(ca);
        let m = 1.0 / ba.dot(p);

        if m.abs() < epsilon {
            return None;
        }

        let oa = ray.origin - vertices[0];
        let u = m * oa.dot(p);

        if u < 0.0 || u > 1.0 {
            return None;
        }

        let q = oa.cross(ba);
        let v = m * ray.velocity.dot(q);

        if v < 0.0 || u + v > 1.0 {
            return None;
        }

        let time = m * ca.dot(q);

        if time < 0.0 || time >= max_time {
            return None;
        }

        Some(TriangleIntersection {
            time: time,
            u: u,
            v: v,
        })
    }
}
