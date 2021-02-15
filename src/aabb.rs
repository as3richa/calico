use crate::tuple::Tuple3;
use crate::Float;

#[derive(Clone, Copy)]
pub struct AABB {
    pub min: Tuple3,
    pub max: Tuple3,
}

impl AABB {
    pub fn new(min: Tuple3, max: Tuple3) -> AABB {
        AABB { min: min, max: max }
    }

    pub fn empty() -> AABB {
        AABB {
            min: Tuple3::new([Float::INFINITY, Float::INFINITY, Float::INFINITY]),
            max: Tuple3::new([-Float::INFINITY, -Float::INFINITY, -Float::INFINITY]),
        }
    }

    pub fn join(self, rhs: AABB) -> AABB {
        AABB {
            min: Tuple3::min(self.min, rhs.min),
            max: Tuple3::max(self.max, rhs.max),
        }
    }

    pub fn add(self, rhs: Tuple3) -> AABB {
        AABB {
            min: Tuple3::min(self.min, rhs),
            max: Tuple3::max(self.max, rhs),
        }
    }

    pub fn centroid(self) -> Tuple3 {
        (self.min + self.max) / 2.0
    }

    pub fn extents(self) -> Tuple3 {
        self.max - self.min
    }
}
