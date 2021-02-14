extern crate calico;

use calico::bvh::{Intersection, Primitive, Ray, AABB, BVH};
use calico::Float;

#[derive(Clone, Copy)]
struct Sphere {
    id: usize,
    x: Float,
    y: Float,
    z: Float,
    r: Float,
}

impl Sphere {
    fn new(id: usize, x: Float, y: Float, z: Float, r: Float) -> Sphere {
        Sphere {
            id: id,
            x: x,
            y: y,
            z: z,
            r: r,
        }
    }
}

impl Primitive<IdTime> for Sphere {
    fn aabb(&self) -> AABB {
        AABB::new(
            [self.x - self.r, self.y - self.r, self.z - self.r],
            [self.x + self.r, self.y + self.r, self.z + self.r],
        )
    }

    fn intersect_first(&self, ray: Ray, max_time: Float) -> Option<IdTime> {
        let p = [
            ray.origin[0] - self.x,
            ray.origin[1] - self.y,
            ray.origin[2] - self.z,
        ];

        let v = ray.velocity;

        let pv = p[0] * v[0] + p[1] * v[1] + p[2] * v[2];
        let pp = p[0] * p[0] + p[1] * p[1] + p[2] * p[2];
        let vv = v[0] * v[0] + v[1] * v[1] + v[2] * v[2];

        let discriminant = pv * pv - pp * vv + self.r * self.r;

        if discriminant < 0.0 {
            None
        } else {
            let radical = Float::sqrt(discriminant);
            let minus = (-pv - radical) / vv;
            let plus = (-pv + radical) / vv;

            if 0.0 <= minus && minus <= max_time {
                Some(IdTime(self.id, minus))
            } else if 0.0 <= plus && plus <= max_time {
                Some(IdTime(self.id, plus))
            } else {
                None
            }
        }
    }
}

#[derive(Debug)]
struct IdTime(usize, Float);

impl Intersection for IdTime {
    fn time(&self) -> Float {
        self.1
    }
}

fn main() {
    let spheres = (0..100)
        .map(|i| {
            (0..100).map(move |j| {
                (0..100).map(move |k| {
                    Sphere::new(
                        i,
                        3.0 * (i as Float),
                        3.0 * (j as Float),
                        3.0 * (k as Float),
                        1.0,
                    )
                })
            })
        })
        .flatten()
        .flatten()
        .collect::<Vec<Sphere>>();
    let bvh = BVH::new(&spheres);
    println!("built");

    for i in 0..50 {
        let y = 18.0 - 20.0 / 50.0 * (i as Float);
        let mut s = String::new();
        for j in 0..100 {
            let x = -2.0 + 20.0 / 100.0 * (j as Float);

            if bvh
                .intersect_first(Ray::new([-1.0, y, x], [1.0, 0.0, 0.0]), Float::INFINITY)
                .is_none()
            {
                s += " ";
            } else {
                s += "x";
            }
        }
        println!("{}|", s);
    }
}
