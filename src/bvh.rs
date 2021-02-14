use crate::Float;
use std::marker::PhantomData;
use std::ops::Range;

pub struct BVH<P: Primitive<I>, I: Intersection> {
    primitives: Vec<P>,
    nodes: Vec<Node>,
    intersection: PhantomData<I>,
}

pub trait Primitive<I: Intersection>: Copy {
    fn aabb(&self) -> AABB;
    fn intersect_first(&self, ray: Ray, max_time: Float) -> Option<I>;
}

pub trait Intersection {
    fn time(&self) -> Float;
}

#[derive(Clone, Copy)]
pub struct Ray {
    pub origin: [Float; 3],
    pub velocity: [Float; 3],
}

impl Ray {
    pub fn new(origin: [Float; 3], velocity: [Float; 3]) -> Ray {
        Ray {
            origin: origin,
            velocity: velocity,
        }
    }
}

#[derive(Clone, Copy)]
pub struct AABB {
    pub min: [Float; 3],
    pub max: [Float; 3],
}

impl AABB {
    pub fn new(min: [Float; 3], max: [Float; 3]) -> AABB {
        AABB { min: min, max: max }
    }

    fn empty() -> AABB {
        AABB {
            min: [Float::INFINITY, Float::INFINITY, Float::INFINITY],
            max: [-Float::INFINITY, -Float::INFINITY, -Float::INFINITY],
        }
    }

    fn extents(self) -> [Float; 3] {
        [
            self.max[0] - self.min[0],
            self.max[1] - self.min[1],
            self.max[2] - self.min[2],
        ]
    }

    fn join(self, rhs: AABB) -> AABB {
        AABB {
            min: [
                Float::min(self.min[0], rhs.min[0]),
                Float::min(self.min[1], rhs.min[1]),
                Float::min(self.min[2], rhs.min[2]),
            ],
            max: [
                Float::max(self.max[0], rhs.max[0]),
                Float::max(self.max[1], rhs.max[1]),
                Float::max(self.max[2], rhs.max[2]),
            ],
        }
    }

    fn add(self, u: [Float; 3]) -> AABB {
        AABB {
            min: [
                Float::min(self.min[0], u[0]),
                Float::min(self.min[1], u[1]),
                Float::min(self.min[2], u[2]),
            ],
            max: [
                Float::max(self.max[0], u[0]),
                Float::max(self.max[1], u[1]),
                Float::max(self.max[2], u[2]),
            ],
        }
    }

    fn centroid(self) -> [Float; 3] {
        [
            (self.min[0] + self.max[0]) / 2.0,
            (self.min[1] + self.max[1]) / 2.0,
            (self.min[2] + self.max[2]) / 2.0,
        ]
    }

    fn half_surface_area(self) -> Float {
        let [w, h, d] = self.extents();
        w * h + w * d + h * d
    }

    fn intersects_ray(
        self,
        origin: [Float; 3],
        inverse_velocity: [Float; 3],
        max_time: Float,
    ) -> bool {
        let mut enters = 0.0;
        let mut exits = max_time;

        for axis in 0..3 {
            let p = origin[axis];
            let inv_v = inverse_velocity[axis];
            let l = self.min[axis];
            let u = self.max[axis];

            let bounds = ((l - p) * inv_v, (u - p) * inv_v);

            let (en, ex) = if inv_v > 0.0 {
                bounds
            } else {
                (bounds.1, bounds.0)
            };

            enters = Float::max(enters, en);
            exits = Float::min(exits, ex);
        }

        debug_assert!(!enters.is_nan());
        debug_assert!(!exits.is_nan());

        enters <= exits * 1.00000024
    }
}

#[derive(Clone)]
enum Node {
    Branch {
        aabb: AABB,
        axis: usize,
        right: usize,
    },
    Leaf {
        aabb: AABB,
        range: Range<usize>,
    },
}

impl Node {
    fn aabb(&self) -> AABB {
        match self {
            Node::Branch {
                aabb,
                axis: _,
                right: _,
            } => *aabb,
            Node::Leaf { aabb, range: _ } => *aabb,
        }
    }
}

struct Info {
    index: usize,
    centroid: [Float; 3],
    aabb: AABB,
}

impl Info {
    fn new(index: usize, aabb: AABB) -> Info {
        Info {
            index: index,
            centroid: aabb.centroid(),
            aabb: aabb,
        }
    }
}

struct Builder<'a, P: Primitive<I>, I: Intersection> {
    primitives: &'a [P],
    info: Vec<Info>,
    reordered: Vec<P>,
    nodes: Vec<Node>,
    intersection: PhantomData<I>,
}

impl<'a, P: Primitive<I>, I: Intersection> Builder<'a, P, I> {
    const TRAVERSAL_COST: Float = 0.125;
    const INTERSECTION_COST: Float = 1.0;

    fn build(primitives: &[P], max_leaf_size: usize) -> BVH<P, I> {
        if primitives.is_empty() {
            let root = Node::Leaf {
                aabb: AABB::empty(),
                range: 0..0,
            };

            return BVH {
                primitives: vec![],
                nodes: vec![root],
                intersection: PhantomData,
            };
        }

        let mut builder = Builder::new(primitives);
        let root = builder.build_recursive(0..primitives.len(), max_leaf_size);
        debug_assert!(root == 0);
        builder.finalize()
    }

    fn new(primitives: &'a [P]) -> Self {
        let len = primitives.len();
        debug_assert!(len > 0);

        let info = primitives
            .iter()
            .enumerate()
            .map(|(i, primitive)| Info::new(i, primitive.aabb()))
            .collect::<Vec<Info>>();

        let reordered = Vec::with_capacity(len);
        let nodes = Vec::with_capacity(2 * len - 1);

        Builder {
            primitives: primitives,
            info: info,
            reordered: reordered,
            nodes: nodes,
            intersection: PhantomData,
        }
    }

    fn build_recursive(&mut self, range: Range<usize>, max_leaf_size: usize) -> usize {
        debug_assert!(!range.is_empty());

        let id = self.allocate_node();

        self.nodes[id] = self
            .choose_axis(range.clone())
            .and_then(|(axis, min, extent)| {
                let (coord, cost) = self.choose_partition(range.clone(), axis, min, extent);
                let leaf_cost = Self::INTERSECTION_COST * (range.len() as Float);

                if range.len() <= max_leaf_size && leaf_cost <= cost {
                    None
                } else {
                    let k = self.partition(range.clone(), axis, coord);
                    let left = self.build_recursive(range.start..k, max_leaf_size);
                    let right = self.build_recursive(k..range.end, max_leaf_size);
                    let aabb = self.nodes[left].aabb().join(self.nodes[right].aabb());
                    debug_assert!(left == id + 1);

                    Some(Node::Branch {
                        aabb: aabb,
                        axis: axis,
                        right: right,
                    })
                }
            })
            .unwrap_or_else(|| {
                let re = self.reordered.len();
                let mut aabb = AABB::empty();

                for info in &self.info[range.clone()] {
                    let primitive = self.primitives[info.index].clone();
                    self.reordered.push(primitive);
                    aabb = aabb.join(primitive.aabb());
                }

                Node::Leaf {
                    aabb: aabb,
                    range: re..re + range.len(),
                }
            });

        id
    }

    fn finalize(self) -> BVH<P, I> {
        let len = self.primitives.len();
        debug_assert!(len > 0);

        debug_assert!(self.reordered.len() == len);

        let mut nodes = self.nodes;
        debug_assert!(nodes.len() <= 2 * len - 1);
        nodes.shrink_to_fit();

        BVH {
            primitives: self.reordered,
            nodes: nodes,
            intersection: PhantomData,
        }
    }

    fn allocate_node(&mut self) -> usize {
        let id = self.nodes.len();
        let placeholder = Node::Leaf {
            aabb: AABB::empty(),
            range: 0..0,
        };
        self.nodes.push(placeholder);
        id
    }

    fn choose_axis(&self, range: Range<usize>) -> Option<(usize, Float, Float)> {
        if range.len() <= 1 {
            return None;
        }

        let aabb = self.info[range]
            .iter()
            .fold(AABB::empty(), |aabb, q| aabb.add(q.centroid));

        let extents = aabb.extents();
        let extent = Float::max(extents[0], Float::max(extents[1], extents[2]));

        if extent < 1e-4 {
            None
        } else {
            let axis = if extents[0] == extent {
                0
            } else if extents[1] == extent {
                1
            } else {
                2
            };

            Some((axis, aabb.min[axis], extent))
        }
    }

    fn choose_partition(
        &self,
        range: Range<usize>,
        axis: usize,
        min: Float,
        extent: Float,
    ) -> (Float, Float) {
        const N_PARTITIONS: usize = 15;
        let mut buckets = [Bucket::new(); N_PARTITIONS];

        for q in &self.info[range] {
            let f = ((q.centroid[axis] - min) / extent) * (N_PARTITIONS as Float);
            let i = usize::min(f as usize, N_PARTITIONS - 1);
            buckets[i] = buckets[i].add(q.aabb);
        }

        let mut suffixes = buckets;
        for i in (0..N_PARTITIONS - 1).rev() {
            suffixes[i] = suffixes[i].join(suffixes[i + 1]);
        }

        let mut best = (0.0, Float::INFINITY);
        let mut prefix = Bucket::new();
        for i in 0..N_PARTITIONS - 1 {
            prefix = prefix.join(buckets[i]);
            let suffix = suffixes[i + 1];

            let cost =
                Self::TRAVERSAL_COST + Self::INTERSECTION_COST * (prefix.cost() + suffix.cost());

            if cost < best.1 {
                best = (prefix.aabb.max[axis], cost);
            }
        }

        debug_assert!(min <= best.0 && best.0 <= min + extent);
        debug_assert!(best.1 < Float::INFINITY);

        best
    }

    fn partition(&mut self, range: Range<usize>, axis: usize, coord: Float) -> usize {
        let mut k = range.start;

        for i in range.clone() {
            if self.info[i].centroid[axis] <= coord {
                self.info.swap(k, i);
                k += 1;
            }
        }

        debug_assert!(range.start < k && k < range.end);
        k
    }
}

#[derive(Clone, Copy)]
struct Bucket {
    aabb: AABB,
    count: usize,
}

impl Bucket {
    fn new() -> Bucket {
        Bucket {
            aabb: AABB::empty(),
            count: 0,
        }
    }

    fn add(self, aabb: AABB) -> Bucket {
        Bucket {
            aabb: self.aabb.join(aabb),
            count: self.count + 1,
        }
    }

    fn join(self, rhs: Bucket) -> Bucket {
        Bucket {
            aabb: self.aabb.join(rhs.aabb),
            count: self.count + rhs.count,
        }
    }

    fn cost(self) -> Float {
        self.aabb.half_surface_area() * (self.count as Float)
    }
}

impl<P: Primitive<I>, I: Intersection> BVH<P, I> {
    pub fn new(primitives: &[P]) -> BVH<P, I> {
        Builder::build(primitives, 8)
    }

    pub fn intersect_first(&self, ray: Ray, mut max_time: Float) -> Option<I> {
        let mut stack = Vec::with_capacity(64);
        stack.push(0);

        let mut first = None;

        let inverse_velocity = [
            1.0 / ray.velocity[0],
            1.0 / ray.velocity[1],
            1.0 / ray.velocity[2],
        ];

        while let Some(mut id) = stack.pop() {
            loop {
                let node = &self.nodes[id];

                if !node
                    .aabb()
                    .intersects_ray(ray.origin, inverse_velocity, max_time)
                {
                    break;
                }

                match node {
                    Node::Branch {
                        aabb: _,
                        axis,
                        right,
                    } => {
                        let left = id + 1;
                        let children = if ray.velocity[*axis] >= 0.0 {
                            [left, *right]
                        } else {
                            [*right, left]
                        };
                        id = children[0];
                        stack.push(children[1]);
                    }

                    Node::Leaf { aabb: _, range } => {
                        for primitive in &self.primitives[range.clone()] {
                            if !primitive.aabb().intersects_ray(
                                ray.origin,
                                inverse_velocity,
                                max_time,
                            ) {
                                continue;
                            }

                            if let Some(intersection) = primitive.intersect_first(ray, max_time) {
                                let time = intersection.time();
                                if time < max_time {
                                    first = Some(intersection);
                                    max_time = time;
                                }
                            }
                        }
                        break;
                    }
                }
            }
        }

        first
    }
}
