use crate::aabb::AABB;
use crate::bvh::Ray;
use crate::tuple::Tuple3;
use crate::Float;

#[derive(Clone, Copy, Debug)]
pub enum Shape {
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

pub struct ShapeIntersection {
    pub time: Float,
    pub normal: Tuple3,
}

impl Shape {
    pub fn aabb(self) -> AABB {
        match self {
            Shape::Sphere => AABB::new(
                Tuple3::new([-1.0, -1.0, -1.0]),
                Tuple3::new([1.0, 1.0, 1.0]),
            ),
        }
    }

    pub fn intersect_first(self, ray: Ray, max_time: Float) -> Option<ShapeIntersection> {
        match self {
            Shape::Sphere => Shape::intersect_unit_sphere(ray, max_time).map(|time| {
                let point = ray.at(time);
                ShapeIntersection {
                    time: time,
                    normal: point.normalize(),
                }
            }),
        }
    }

    pub fn intersect_pred(self, ray: Ray, max_time: Float) -> bool {
        match self {
            Shape::Sphere => Shape::intersect_unit_sphere(ray, max_time).is_some(),
        }
    }

    pub fn intersect_unit_sphere(ray: Ray, max_time: Float) -> Option<Float> {
        let oo = ray.origin.dot(ray.origin);
        let ov = ray.origin.dot(ray.velocity);
        let vv = ray.velocity.dot(ray.velocity);
        let radicand = ov * ov - vv * (oo - 1.0);

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
}
