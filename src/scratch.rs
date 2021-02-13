use crate::Float;
use std::ops;

#[derive(Clone, Copy)]
struct Tuple3 {
    x: Float,
    y: Float,
    z: Float,
}

impl Tuple3 {
    fn new(x: Float, y: Float, z: Float) -> Tuple3 {
        Tuple3 { x: x, y: y, z: z }
    }

    fn cross(self, rhs: Tuple3) -> Tuple3 {
        Tuple3::new(
            self.y * rhs.z - self.z * rhs.y,
            self.z * rhs.x - self.x * rhs.z,
            self.x * rhs.y - self.y * rhs.x,
        )
    }
}

impl ops::Sub<Tuple3> for Tuple3 {
    type Output = Tuple3;

    fn sub(self, rhs: Tuple3) -> Tuple3 {
        Tuple3::new(self.x - rhs.x, self.y - rhs.y, self.z - rhs.z)
    }
}

struct Tuple2 {
    x: Float,
    y: Float,
}

impl Tuple2 {
    fn new(x: Float, y: Float) -> Tuple2 {
        Tuple2 { x: x, y: y }
    }
}

struct TriangleMesh {
    vertices: Vec<Tuple3>,
    vertex_textures: Vec<Tuple2>,
    vertex_normals: Vec<Tuple3>,
    triangles: Vec<[(usize, usize, usize); 3]>,
}

enum TriangleMeshIndexOutOfRange {
    Vertex,
    Texture,
    Normal,
}

impl TriangleMesh {
    fn new() -> TriangleMesh {
        TriangleMesh {
            vertices: vec![],
            vertex_textures: vec![
                Tuple2::new(0.0, 0.0),
                Tuple2::new(1.0, 0.0),
                Tuple2::new(0.0, 1.0),
            ],
            vertex_normals: vec![],
            triangles: vec![],
        }
    }
    fn vertex(&mut self, p: Tuple3) -> usize {
        self.vertices.push(p);
        self.vertices.len() - 1
    }

    fn vertex_texture(&mut self, uv: Tuple2) -> usize {
        self.vertex_textures.push(uv);
        self.vertex_textures.len() - 1
    }

    fn vertex_normal(&mut self, n: Tuple3) -> usize {
        self.vertex_normals.push(n);
        self.vertex_normals.len() - 1
    }

    fn triangle(
        &mut self,
        vertices: [usize; 3],
        textures: Option<[usize; 3]>,
        normals: Option<[usize; 3]>,
    ) -> Result<(), TriangleMeshIndexOutOfRange> {
        fn check(
            ary: [usize; 3],
            count: usize,
            error: TriangleMeshIndexOutOfRange,
        ) -> Result<(), TriangleMeshIndexOutOfRange> {
            if ary.iter().all(|&x| x < count) {
                Ok(())
            } else {
                Err(error)
            }
        };

        check(
            vertices,
            self.vertices.len(),
            TriangleMeshIndexOutOfRange::Vertex,
        )?;

        let unwrapped_textures = match textures {
            Some(textures) => {
                check(
                    textures,
                    self.vertex_textures.len(),
                    TriangleMeshIndexOutOfRange::Texture,
                )?;
                textures
            }
            None => [0, 1, 2],
        };

        let unwrapped_normals = match normals {
            Some(normals) => {
                check(
                    normals,
                    self.vertex_normals.len(),
                    TriangleMeshIndexOutOfRange::Normal,
                )?;
                normals
            }
            None => {
                let p = self.vertices[vertices[0]];
                let q = self.vertices[vertices[1]];
                let r = self.vertices[vertices[2]];
                let ni = self.vertex_normal((p - r).cross(q - r));
                [ni; 3]
            }
        };

        self.triangles.push([
            (vertices[0], unwrapped_textures[0], unwrapped_normals[0]),
            (vertices[1], unwrapped_textures[1], unwrapped_normals[1]),
            (vertices[2], unwrapped_textures[2], unwrapped_normals[2]),
        ]);

        Ok(())
    }
}
