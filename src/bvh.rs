use crate::aabb::AABB;
use crate::tuple::Tuple3;
use crate::Float;
use std::marker::PhantomData;
use std::ops::Range;

pub struct BVH<P: Primitive<I>, I: Intersection> {
    primitives: Vec<P>,
    nodes: Vec<Node>,
    intersection: PhantomData<I>,
}

pub trait Primitive<I: Intersection>: Clone {
    fn aabb(&self) -> AABB;
    fn intersect_first(&self, ray: Ray, max_time: Float) -> Option<I>;
}

pub trait Intersection {
    fn time(&self) -> Float;
}

#[derive(Clone, Copy, Debug)]
pub struct Ray {
    pub origin: Tuple3,
    pub velocity: Tuple3,
}

impl Ray {
    pub fn new(origin: Tuple3, velocity: Tuple3) -> Ray {
        Ray {
            origin: origin,
            velocity: velocity,
        }
    }

    pub fn at(self, time: Float) -> Tuple3 {
        self.origin + self.velocity * time
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
    centroid: Tuple3,
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
                    aabb = aabb.join(primitive.aabb());
                    self.reordered.push(primitive);
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

        let (axis, extent) = aabb.extents().argmax();

        if extent < 1e-4 {
            // FIXME: epsilon?
            None
        } else {
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
            buckets[i] = buckets[i].add(q.aabb, q.centroid[axis]);
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
                best = (prefix.max_position, cost);
            }
        }

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

#[derive(Clone, Copy, Debug)]
struct Bucket {
    aabb: AABB,
    max_position: Float,
    count: usize,
}

impl Bucket {
    fn new() -> Bucket {
        Bucket {
            aabb: AABB::empty(),
            max_position: -Float::INFINITY,
            count: 0,
        }
    }

    fn add(self, aabb: AABB, position: Float) -> Bucket {
        Bucket {
            aabb: self.aabb.join(aabb),
            max_position: Float::max(self.max_position, position),
            count: self.count + 1,
        }
    }

    fn join(self, rhs: Bucket) -> Bucket {
        Bucket {
            aabb: self.aabb.join(rhs.aabb),
            max_position: Float::max(self.max_position, rhs.max_position),
            count: self.count + rhs.count,
        }
    }

    fn cost(self) -> Float {
        let extents = self.aabb.extents();
        let half_surface_area =
            extents.x() * extents.y() + extents.x() * extents.z() + extents.y() * extents.z();
        half_surface_area * (self.count as Float)
    }
}

fn aabb_ray_robust_intersect(
    aabb: AABB,
    origin: Tuple3,
    inv_velocity: Tuple3,
    max_time: Float,
) -> bool {
    let mut enters = 0.0;
    let mut exits = max_time;

    for axis in 0..3 {
        let o = origin[axis];
        let iv = inv_velocity[axis];
        let l = aabb.min[axis];
        let u = aabb.max[axis];

        let bounds = ((l - o) * iv, (u - o) * iv);

        let (en, ex) = if iv > 0.0 {
            bounds
        } else {
            (bounds.1, bounds.0)
        };

        enters = Float::max(enters, en);
        exits = Float::min(exits, ex);
    }

    debug_assert!(!enters.is_nan());
    debug_assert!(!exits.is_nan());

    // ref. Thiago Ize, Robust BVH Ray Traversal (http://jcgt.org/published/0002/02/02/)
    enters <= exits * 1.00000024
}

impl<P: Primitive<I>, I: Intersection> BVH<P, I> {
    pub fn new(primitives: &[P]) -> BVH<P, I> {
        Builder::build(primitives, 8)
    }

    pub fn intersect_first(&self, ray: Ray, mut max_time: Float) -> Option<I> {
        let mut stack = Vec::with_capacity(64);
        stack.push(0);

        let mut first = None;

        let inv_velocity = Tuple3::new([
            1.0 / ray.velocity[0],
            1.0 / ray.velocity[1],
            1.0 / ray.velocity[2],
        ]);

        while let Some(mut id) = stack.pop() {
            loop {
                let node = &self.nodes[id];

                if !aabb_ray_robust_intersect(node.aabb(), ray.origin, inv_velocity, max_time) {
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
                            if !aabb_ray_robust_intersect(
                                primitive.aabb(),
                                ray.origin,
                                inv_velocity,
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
