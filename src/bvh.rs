use crate::Float;
use std::ops::Range;

#[derive(Clone, Copy)]
struct BoundingBox {
    min: [Float; 3],
    max: [Float; 3],
}

impl BoundingBox {
    fn empty() -> BoundingBox {
        BoundingBox {
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

    fn join(self, rhs: BoundingBox) -> BoundingBox {
        BoundingBox {
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

    fn add(self, u: [Float; 3]) -> BoundingBox {
        BoundingBox {
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
}

trait Primitive: Copy {
    fn bounding_box(&self) -> BoundingBox;
}

struct BoundingVolumeHierarchy<P: Primitive> {
    primitives: Vec<P>,
    nodes: Vec<Node>,
}

#[derive(Clone)]
enum Node {
    Branch {
        bb: BoundingBox,
        axis: usize,
        right: usize,
    },
    Leaf {
        bb: BoundingBox,
        range: Range<usize>,
    },
}

impl Node {
    fn bb(&self) -> BoundingBox {
        match self {
            Node::Branch {
                bb,
                axis: _,
                right: _,
            } => *bb,
            Node::Leaf { bb, range: _ } => *bb,
        }
    }
}

struct Info {
    index: usize,
    centroid: [Float; 3],
    bb: BoundingBox,
}

impl Info {
    fn new<P: Primitive>(index: usize, primitive: &P) -> Info {
        let bb = primitive.bounding_box();
        Info {
            index: index,
            centroid: bb.centroid(),
            bb: bb,
        }
    }
}

struct Builder<'a, P: Primitive> {
    primitives: &'a [P],
    info: Vec<Info>,
    reordered: Vec<P>,
    nodes: Vec<Node>,
    node_allocator: usize,
}

impl<'a, P: Primitive> Builder<'a, P> {
    const N_PARTITIONS: usize = 15;
    const SAH_T_TRAV: Float = 1.0 / 8.0;
    const SAH_T_ISECT: Float = 1.0;

    fn build(primitives: &'a [P]) -> BoundingVolumeHierarchy<P> {
        if primitives.is_empty() {
            return BoundingVolumeHierarchy {
                primitives: vec![],
                nodes: vec![Self::empty_node()],
            };
        }

        let builder = Self::new(primitives);

        unimplemented!();
    }

    fn new(primitives: &'a [P]) -> Builder<P> {
        let len = primitives.len();

        let mut info = primitives
            .iter()
            .enumerate()
            .map(|(i, primitive)| Info::new(i, primitive))
            .collect::<Vec<Info>>();

        let mut reordered = Vec::with_capacity(len);

        let max_nodes = 2 * len - 1;
        let nodes = vec![Self::empty_node(); max_nodes];

        Builder {
            primitives: primitives,
            info: info,
            reordered: reordered,
            nodes: nodes,
            node_allocator: 0,
        }
    }

    fn empty_node() -> Node {
        Node::Leaf {
            bb: BoundingBox::empty(),
            range: 0..0,
        }
    }

    fn build_recursive(&mut self, range: Range<usize>, max_leaf_size: usize) -> usize {
        debug_assert!(!range.is_empty());

        let id = self.allocate_node();

        self.nodes[id] = self
            .choose_axis(range.clone())
            .and_then(|(axis, min, extent)| {
                let (coord, cost) = self.choose_partition(range.clone(), axis, min, extent);
                let leaf_cost = Self::SAH_T_ISECT * (range.len() as Float);

                if range.len() <= max_leaf_size && leaf_cost <= cost {
                    None
                } else {
                    let k = self.partition(range.clone(), axis, coord);
                    let left = self.build_recursive(range.start..k, max_leaf_size);
                    let right = self.build_recursive(k..range.end, max_leaf_size);
                    let bb = self.nodes[left].bb().join(self.nodes[right].bb());
                    debug_assert!(left == id + 1);

                    Some(Node::Branch {
                        bb: bb,
                        axis: axis,
                        right: right,
                    })
                }
            })
            .unwrap_or_else(|| {
                let re = self.reordered.len();
                let mut bb = BoundingBox::empty();

                for primitive in &self.primitives[range.clone()] {
                    self.reordered.push(primitive.clone());
                    bb = bb.join(primitive.bounding_box());
                }

                Node::Leaf {
                    bb: bb,
                    range: re..re + range.len(),
                }
            });

        id
    }

    fn allocate_node(&mut self) -> usize {
        let id = self.node_allocator;
        self.node_allocator += 1;
        debug_assert!(self.node_allocator <= 2 * self.primitives.len() - 1);
        id
    }

    fn choose_axis(&self, range: Range<usize>) -> Option<(usize, Float, Float)> {
        if range.len() <= 1 {
            return None;
        }

        let bb = self.info[range]
            .iter()
            .fold(BoundingBox::empty(), |bb, q| bb.add(q.centroid));

        let extents = bb.extents();
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

            Some((axis, bb.min[axis], extent))
        }
    }

    fn choose_partition(
        &self,
        range: Range<usize>,
        axis: usize,
        min: Float,
        extent: Float,
    ) -> (Float, Float) {
        const N_BUCKETS: usize = 15;
        let mut buckets = [Bucket::new(); N_BUCKETS];

        for q in &self.info[range] {
            let f = ((q.centroid[axis] - min) / extent) * (N_BUCKETS as Float);
            let i = usize::min(f as usize, N_BUCKETS - 1);
            buckets[i] = buckets[i].add(q.bb);
        }

        let mut suffixes = buckets;
        for i in (0..N_BUCKETS - 1).rev() {
            suffixes[i] = suffixes[i].join(suffixes[i + 1]);
        }

        let mut best = (Float::INFINITY, 0.0);
        let mut prefix = Bucket::new();
        for i in 0..N_BUCKETS - 1 {
            prefix = prefix.join(buckets[i]);
            let suffix = suffixes[i + 1];
            let cost = Self::SAH_T_TRAV + Self::SAH_T_ISECT * (prefix.cost() + suffix.cost());
            if cost < best.0 {
                best = (cost, prefix.bb.max[axis]);
            }
        }

        (best.0, best.1)
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
    bb: BoundingBox,
    count: usize,
}

impl Bucket {
    fn new() -> Bucket {
        Bucket {
            bb: BoundingBox::empty(),
            count: 0,
        }
    }

    fn add(self, bb: BoundingBox) -> Bucket {
        Bucket {
            bb: self.bb.join(bb),
            count: self.count + 1,
        }
    }

    fn join(self, rhs: Bucket) -> Bucket {
        Bucket {
            bb: self.bb.join(rhs.bb),
            count: self.count + rhs.count,
        }
    }

    fn cost(self) -> Float {
        self.bb.half_surface_area() * (self.count as Float)
    }
}
/*
impl<P: Primitive> BoundingVolumeHierarchy<P> {
    fn new(primitives: &[P], max_leaf_size: usize) -> BoundingVolumeHierarchy<P> {
        if primitives.is_empty() {
            let root = Node::Leaf {
                bb: BoundingBox::empty(),
                primitives: 0..0,
            };

            return BoundingVolumeHierarchy {
                primitives: vec![],
                nodes: vec![root],
            };
        }

        let len = primitives.len();
        let max_nodes = 2 * len - 1;

        let mut reordered = Vec::with_capacity(len);
        let mut nodes = Vec::with_capacity(max_nodes);
        let mut info = primitives
            .iter()
            .enumerate()
            .map(|(i, primitive)| Info::new(i, primitive))
            .collect::<Vec<Info>>();

        let mut build = |begin, end| {
            debug_assert!(begin < end);

            let info = &mut info[begin..end];

            let (axis, min, extent) = Self::choose_axis(info);
            let coord = Self::choose_partition(info, axis, min, extent);

            let mut k = 0;

            for i in 0..info.len() {
                if info[i].centroid[axis] <= coord {
                    info.swap(i, k);
                    k += 1;
                }
            }

            debug_assert!(k != 0 && k != info.len());

            build(begin, begin + k);
            build()
        };

        build(0, len);

        debug_assert!(reordered.len() == len);
        debug_assert!(nodes.len() <= max_nodes);

        nodes.shrink_to_fit();

        BoundingVolumeHierarchy {
            primitives: reordered,
            nodes: nodes,
        }
    }
}
*/
