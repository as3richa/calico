use crate::bvh::Ray;
use crate::tuple::{Tuple, Tuple3};
use crate::Float;
use std::ops;

#[cfg(test)]
use crate::eq_approx;

#[derive(Clone, Debug)]
pub struct Matrix([[Float; 4]; 4]);

impl Matrix {
    pub fn new(xs: [[Float; 4]; 4]) -> Matrix {
        Matrix(xs)
    }

    pub fn identity() -> Matrix {
        Matrix([
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ])
    }

    pub fn translation(x: Float, y: Float, z: Float) -> Matrix {
        Matrix([
            [1.0, 0.0, 0.0, x],
            [0.0, 1.0, 0.0, y],
            [0.0, 0.0, 1.0, z],
            [0.0, 0.0, 0.0, 1.0],
        ])
    }

    pub fn scaling(x: Float, y: Float, z: Float) -> Matrix {
        Matrix([
            [x, 0.0, 0.0, 0.0],
            [0.0, y, 0.0, 0.0],
            [0.0, 0.0, z, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ])
    }

    pub fn rotation_x(angle: Float) -> Matrix {
        Matrix::rotation(angle, 1, 2)
    }

    pub fn rotation_y(angle: Float) -> Matrix {
        Matrix::rotation(angle, 2, 0)
    }

    pub fn rotation_z(angle: Float) -> Matrix {
        Matrix::rotation(angle, 0, 1)
    }

    fn rotation(angle: Float, right: usize, up: usize) -> Matrix {
        let cos = Float::cos(angle);
        let sin = Float::sin(angle);

        let mut xs = [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ];
        xs[right][right] = cos;
        xs[right][up] = -sin;
        xs[up][up] = cos;
        xs[up][right] = sin;

        Matrix(xs)
    }

    pub fn shearing(xy: Float, xz: Float, yx: Float, yz: Float, zx: Float, zy: Float) -> Matrix {
        Matrix([
            [1.0, xy, xz, 0.0],
            [yx, 1.0, yz, 0.0],
            [zx, zy, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ])
    }

    pub fn look_at(eye: Tuple3, center: Tuple3, up: Tuple3) -> Matrix {
        let forward = (center - eye).normalize();
        let right = up.cross(forward).normalize();
        let up_perp = forward.cross(right);

        Matrix::new([
            [right[0], right[1], right[2], -eye.dot(right)],
            [up_perp[0], up_perp[1], up_perp[2], -eye.dot(up_perp)],
            [forward[0], forward[1], forward[2], -eye.dot(forward)],
            [0.0, 0.0, 0.0, 1.0],
        ])
    }

    pub fn transpose(&mut self) -> &mut Matrix {
        let mut swap = |(i1, j1), (i2, j2)| {
            let x = self[i1][j1];
            self[i1][j1] = self[i2][j2];
            self[i2][j2] = x;
        };
        swap((0, 1), (1, 0));
        swap((0, 2), (2, 0));
        swap((0, 3), (3, 0));
        swap((1, 2), (2, 1));
        swap((1, 3), (3, 1));
        swap((2, 3), (3, 2));
        self
    }

    pub fn mul(&mut self, rhs: &Matrix) {
        let mut mul_row = |i| {
            let row = self[i];
            for j in 0..4 {
                self[i][j] = 0.0;
                for k in 0..4 {
                    self[i][j] += row[k] * rhs[k][j];
                }
            }
        };

        for i in 0..4 {
            mul_row(i);
        }
    }

    pub fn determinant(&self) -> Float {
        self[0][0] * self.cofactor(0, 0)
            + self[0][1] * self.cofactor(0, 1)
            + self[0][2] * self.cofactor(0, 2)
            + self[0][3] * self.cofactor(0, 3)
    }

    // FIXME: inverse?
    pub fn invert(&mut self) -> bool {
        match self.inverse() {
            Some(m) => {
                *self = m;
                true
            }
            None => false,
        }
    }

    pub fn inverse(&self) -> Option<Matrix> {
        let det = self.determinant();

        // FIXME: epsilon?
        if det.abs() < 1e-3 {
            return None;
        }

        let mut m = [[0.0; 4]; 4];

        for i in 0..4 {
            for j in 0..4 {
                m[j][i] = self.cofactor(i, j) / det;
            }
        }

        Some(Matrix::new(m))
    }

    fn minor3x3(
        &self,
        row0: usize,
        row1: usize,
        row2: usize,
        col0: usize,
        col1: usize,
        col2: usize,
    ) -> Float {
        det3x3(
            self[row0][col0],
            self[row0][col1],
            self[row0][col2],
            self[row1][col0],
            self[row1][col1],
            self[row1][col2],
            self[row2][col0],
            self[row2][col1],
            self[row2][col2],
        )
    }

    fn cofactor(&self, i: usize, j: usize) -> Float {
        let indices = |k| {
            if k == 0 {
                (1, 2, 3)
            } else if k == 1 {
                (0, 2, 3)
            } else if k == 2 {
                (0, 1, 3)
            } else {
                (0, 1, 2)
            }
        };

        let (row0, row1, row2) = indices(i);
        let (col0, col1, col2) = indices(j);

        let minor = self.minor3x3(row0, row1, row2, col0, col1, col2);

        if (i + j) % 2 == 0 {
            minor
        } else {
            -minor
        }
    }

    pub fn translate(&mut self, x: Float, y: Float, z: Float) {
        let mut translate_row = |i, x| {
            for j in 0..4 {
                self[i][j] += self[3][j] * x;
            }
        };
        translate_row(0, x);
        translate_row(1, y);
        translate_row(2, z);
    }

    pub fn scale(&mut self, x: Float, y: Float, z: Float) {
        let mut scale_row = |i, x| {
            for j in 0..4 {
                self[i][j] *= x;
            }
        };
        scale_row(0, x);
        scale_row(1, y);
        scale_row(2, z);
    }

    pub fn rotate_x(&mut self, angle: Float) {
        self.rotate(angle, 1, 2);
    }

    pub fn rotate_y(&mut self, angle: Float) {
        self.rotate(angle, 2, 0);
    }

    pub fn rotate_z(&mut self, angle: Float) {
        self.rotate(angle, 0, 1);
    }

    fn rotate(&mut self, angle: Float, right: usize, up: usize) {
        let cos = Float::cos(angle);
        let sin = Float::sin(angle);

        let row = self[right];

        for j in 0..4 {
            self[right][j] = cos * row[j] - sin * self[up][j];
        }

        for j in 0..4 {
            self[up][j] = sin * row[j] + cos * self[up][j];
        }
    }

    pub fn shear(&mut self, xy: Float, xz: Float, yx: Float, yz: Float, zx: Float, zy: Float) {
        let row0 = self[0];
        let row1 = self[1];

        for j in 0..4 {
            self[0][j] = row0[j] + xy * row1[j] + xz * self[2][j];
            self[1][j] = yx * row0[j] + row1[j] + yz * self[2][j];
        }

        for j in 0..4 {
            self[2][j] = zx * row0[j] + zy * row1[j] + self[2][j];
        }
    }

    pub fn transform_ray(&self, ray: Ray) -> Ray {
        Ray::new(
            (self * ray.origin.as_point()).as_tuple3(),
            (self * ray.velocity.as_vector()).as_tuple3(),
        )
    }

    #[cfg(test)]
    fn eq_approx(&self, rhs: &Matrix) -> bool {
        (0..4).all(|i| (0..4).all(|j| eq_approx(self[i][j], rhs[i][j])))
    }
}

fn det2x2(a: Float, b: Float, c: Float, d: Float) -> Float {
    a * d - b * c
}

fn det3x3(
    a: Float,
    b: Float,
    c: Float,
    d: Float,
    e: Float,
    f: Float,
    g: Float,
    h: Float,
    i: Float,
) -> Float {
    a * det2x2(e, f, h, i) - b * det2x2(d, f, g, i) + c * det2x2(d, e, g, h)
}

impl ops::Mul<Tuple> for &Matrix {
    type Output = Tuple;

    fn mul(self, u: Tuple) -> Tuple {
        let mul_row = |i| u.x * self[i][0] + u.y * self[i][1] + u.z * self[i][2] + u.w * self[i][3];
        Tuple::new(mul_row(0), mul_row(1), mul_row(2), mul_row(3))
    }
}

impl ops::Index<usize> for Matrix {
    type Output = [Float; 4];

    fn index(&self, row: usize) -> &[Float; 4] {
        &self.0[row]
    }
}

impl ops::IndexMut<usize> for Matrix {
    fn index_mut(&mut self, row: usize) -> &mut [Float; 4] {
        &mut self.0[row]
    }
}

#[cfg(test)]
mod tests {
    use super::Matrix;
    use crate::finite::Finite;
    use crate::tuple::tests::{Point, Vector};
    use crate::tuple::Tuple;
    use crate::{consts, eq_approx, Float};
    use quickcheck::TestResult;
    use quickcheck::{Arbitrary, Gen};

    impl Arbitrary for Matrix {
        fn arbitrary(gen: &mut Gen) -> Matrix {
            let mut xs = [[0.0; 4]; 4];
            for i in 0..4 {
                for j in 0..4 {
                    let x = loop {
                        let Finite(x) = Arbitrary::arbitrary(gen);
                        if x.abs() <= 1e3 {
                            break x;
                        }
                    };
                    xs[i][j] = x;
                }
            }
            Matrix(xs)
        }
    }

    #[derive(Clone, Debug)]
    struct DiagonalMatrix(Matrix);

    impl Arbitrary for DiagonalMatrix {
        fn arbitrary(gen: &mut Gen) -> DiagonalMatrix {
            let Finite(x): Finite = Arbitrary::arbitrary(gen);
            let Finite(y): Finite = Arbitrary::arbitrary(gen);
            let Finite(z): Finite = Arbitrary::arbitrary(gen);
            let Finite(w): Finite = Arbitrary::arbitrary(gen);

            DiagonalMatrix(Matrix::new([
                [x, 0.0, 0.0, 0.0],
                [0.0, y, 0.0, 0.0],
                [0.0, 0.0, z, 0.0],
                [0.0, 0.0, 0.0, w],
            ]))
        }
    }

    #[quickcheck]
    fn new(Matrix(xs): Matrix) -> bool {
        let m = Matrix::new(xs);
        (0..4).all(|i| (0..4).all(|j| eq_approx(m[i][j], xs[i][j])))
    }

    #[quickcheck]
    fn mul() -> bool {
        let m = Matrix::new([
            [1.0, 2.0, 3.0, 4.0],
            [5.0, 6.0, 7.0, 8.0],
            [9.0, 10.0, 11.0, 12.0],
            [13.0, 14.0, 15.0, 16.0],
        ]);

        let m2 = Matrix::new([
            [16.0, 15.0, 14.0, 13.0],
            [12.0, 11.0, 10.0, 9.0],
            [8.0, 7.0, 6.0, 5.0],
            [4.0, 3.0, 2.0, 1.0],
        ]);

        let mut m3 = m.clone();
        m3.mul(&m2);

        m3.eq_approx(&Matrix::new([
            [80.0, 70.0, 60.0, 50.0],
            [240.0, 214.0, 188.0, 162.0],
            [400.0, 358.0, 316.0, 274.0],
            [560.0, 502.0, 444.0, 386.0],
        ]))
    }

    #[quickcheck]
    fn mul_identity(m: Matrix) -> bool {
        let mut m2 = m.clone();
        m2.mul(&Matrix::identity());

        let mut m3 = Matrix::identity();
        m3.mul(&m);

        m2.eq_approx(&m) && m3.eq_approx(&m)
    }

    #[quickcheck]
    fn mul_transpose(m: Matrix, m2: Matrix) -> bool {
        let mut m3 = m.clone();
        m3.mul(&m2);
        m3.transpose();

        let mut m4 = m2.clone();
        let mut m5 = m.clone();
        m4.transpose();
        m5.transpose();
        m4.mul(&m5);

        m3.eq_approx(&m4)
    }

    #[quickcheck]
    fn transpose(m: Matrix) -> bool {
        let mut m2 = m.clone();
        m2.transpose();
        (0..4).all(|i| (0..4).all(|j| eq_approx(m[i][j], m2[j][i])))
    }

    #[quickcheck]
    fn transpose_involutive(m: Matrix) -> bool {
        let mut m2 = m.clone();
        m2.transpose();
        m2.transpose();
        m.eq_approx(&m2)
    }

    #[quickcheck]
    fn mul_tuple() -> bool {
        let m = Matrix::new([
            [1.0, 2.0, 3.0, 4.0],
            [5.0, 6.0, 7.0, 8.0],
            [9.0, 10.0, 11.0, 12.0],
            [13.0, 14.0, 15.0, 16.0],
        ]);
        (&m * Tuple::new(17.0, 18.0, 19.0, 20.0)).eq_approx(Tuple::new(190.0, 486.0, 782.0, 1078.0))
    }

    #[quickcheck]
    fn mul_tuple_identity(u: Tuple) -> bool {
        (&Matrix::identity() * u).eq_approx(u)
    }

    #[quickcheck]
    fn determinant() -> bool {
        let m = Matrix::new([
            [1.0, 2.0, 3.0, 4.0],
            [5.0, 6.0, 7.0, 8.0],
            [9.0, 10.0, 11.0, 12.0],
            [13.0, 14.0, 15.0, 16.0],
        ]);

        let m2 = Matrix::new([
            [-2.0, -8.0, 3.0, 5.0],
            [-3.0, 1.0, 7.0, 3.0],
            [1.0, 2.0, -9.0, 6.0],
            [-6.0, 7.0, 7.0, -9.0],
        ]);

        eq_approx(m.determinant(), 0.0) && eq_approx(m2.determinant(), -4071.0)
    }

    #[quickcheck]
    fn determinant_diagonal(DiagonalMatrix(m): DiagonalMatrix) -> TestResult {
        let expectation = m[0][0] * m[1][1] * m[2][2] * m[3][3];
        if expectation.is_infinite() {
            TestResult::discard()
        } else {
            TestResult::from_bool(eq_approx(m.determinant(), expectation))
        }
    }

    #[quickcheck]
    fn invert() -> bool {
        let mut examples = [
            (
                Matrix::new([
                    [-5.0, 2.0, 6.0, -8.0],
                    [1.0, -5.0, 1.0, 8.0],
                    [7.0, 7.0, -6.0, -7.0],
                    [1.0, -3.0, 7.0, 4.0],
                ]),
                Matrix::new([
                    [0.21804511, 0.45112782, 0.2406015, -0.04511278],
                    [-0.80827068, -1.45676692, -0.44360902, 0.52067669],
                    [-0.07894737, -0.22368421, -0.05263158, 0.19736842],
                    [-0.52255639, -0.81390977, -0.30075188, 0.30639098],
                ]),
            ),
            (
                Matrix::new([
                    [8.0, -5.0, 9.0, 2.0],
                    [7.0, 5.0, 6.0, 1.0],
                    [-6.0, 0.0, 9.0, 6.0],
                    [-3.0, 0.0, -9.0, -4.0],
                ]),
                Matrix::new([
                    [-0.15384615, -0.15384615, -0.28205128, -0.53846154],
                    [-0.07692308, 0.12307692, 0.02564103, 0.03076923],
                    [0.35897436, 0.35897436, 0.43589744, 0.92307692],
                    [-0.69230769, -0.69230769, -0.76923077, -1.92307692],
                ]),
            ),
            (
                Matrix::new([
                    [9.0, 3.0, 0.0, 9.0],
                    [-5.0, -2.0, -6.0, -3.0],
                    [-4.0, 9.0, 6.0, 4.0],
                    [-7.0, 6.0, 6.0, 2.0],
                ]),
                Matrix::new([
                    [-0.04074074, -0.07777778, 0.14444444, -0.22222222],
                    [-0.07777778, 0.03333333, 0.36666667, -0.33333333],
                    [-0.02901235, -0.1462963, -0.10925926, 0.12962963],
                    [0.17777778, 0.06666667, -0.26666667, 0.33333333],
                ]),
            ),
            (Matrix::identity(), Matrix::identity()),
        ];
        examples
            .iter_mut()
            .all(|(m, inv)| m.invert() && m.eq_approx(inv))
    }

    #[quickcheck]
    fn translation(
        Finite(x): Finite,
        Finite(y): Finite,
        Finite(z): Finite,
        Vector(u): Vector,
        Point(p): Point,
    ) -> bool {
        let m = Matrix::translation(x, y, z);
        (&m * u).eq_approx(u) && (&m * p).eq_approx(p + Tuple::vector(x, y, z))
    }

    #[quickcheck]
    fn translation_invert(Finite(x): Finite, Finite(y): Finite, Finite(z): Finite) -> bool {
        let mut m = Matrix::translation(x, y, z);
        m.invert() && m.eq_approx(&Matrix::translation(-x, -y, -z))
    }

    #[quickcheck]
    fn translate(m: Matrix, Finite(x): Finite, Finite(y): Finite, Finite(z): Finite) -> bool {
        let mut m2 = m.clone();
        m2.translate(x, y, z);

        let mut m3 = Matrix::translation(x, y, z);
        m3.mul(&m);

        m2.eq_approx(&m3)
    }

    fn scale_tuple(u: Tuple, x: Float, y: Float, z: Float) -> Tuple {
        Tuple::new(u.x * x, u.y * y, u.z * z, u.w)
    }

    #[quickcheck]
    fn scaling(
        Finite(x): Finite,
        Finite(y): Finite,
        Finite(z): Finite,
        Vector(u): Vector,
        Point(p): Point,
    ) -> bool {
        let m = Matrix::scaling(x, y, z);
        (&m * u).eq_approx(scale_tuple(u, x, y, z)) && (&m * p).eq_approx(scale_tuple(p, x, y, z))
    }

    #[quickcheck]
    fn scaling_invert(Finite(x): Finite, Finite(y): Finite, Finite(z): Finite) -> bool {
        let mut m = Matrix::scaling(x, y, z);
        if !m.invert() {
            eq_approx(x * y * z, 0.0)
        } else {
            m.eq_approx(&Matrix::scaling(1.0 / x, 1.0 / y, 1.0 / z))
        }
    }

    #[quickcheck]
    fn scale(m: Matrix, Finite(x): Finite, Finite(y): Finite, Finite(z): Finite) -> bool {
        let mut m2 = m.clone();
        m2.scale(x, y, z);

        let mut m3 = Matrix::scaling(x, y, z);
        m3.mul(&m);

        m2.eq_approx(&m3)
    }

    fn rotation<F: Fn(Float) -> Matrix, G: Fn(Tuple) -> Tuple>(
        rotation: F,
        rotate_quarter: G,
    ) -> bool {
        let none = rotation(0.0);
        let quarter = rotation(consts::FRAC_PI_2);
        let half = rotation(consts::PI);
        let three_quarter = rotation(3.0 * consts::FRAC_PI_2);
        let full = rotation(2.0 * consts::PI);

        let check = |&u| {
            (&none * u).eq_approx(u)
                && (&quarter * u).eq_approx(rotate_quarter(u))
                && (&half * u).eq_approx(rotate_quarter(rotate_quarter(u)))
                && (&three_quarter * u).eq_approx(rotate_quarter(rotate_quarter(rotate_quarter(u))))
                && (&full * u).eq_approx(u)
        };

        let examples = [
            Tuple::point(1.0, 0.0, 0.0),
            Tuple::point(0.0, 1.0, 0.0),
            Tuple::point(0.0, 0.0, 1.0),
            Tuple::point(-26.57, 45.08, -29.26),
            Tuple::point(32.38, 12.02, -15.91),
            Tuple::point(-12.74, -42.49, -24.19),
            Tuple::point(1.22, -18.32, -10.53),
            Tuple::vector(-9.08, 22.38, 42.64),
            Tuple::point(-6.46, -23.97, -47.29),
            Tuple::point(-48.82, 46.2, 45.07),
            Tuple::new(39.51, -48.59, -11.33, -9.26),
            Tuple::new(-16.26, -6.15, -29.38, -16.02),
            Tuple::point(12.75, 5.13, -6.41),
            Tuple::vector(-40.83, -37.8, 0.19),
            Tuple::vector(48.47, 34.38, -17.97),
            Tuple::new(22.98, 39.59, -9.86, 45.69),
            Tuple::new(-19.99, -0.39, -17.18, -14.44),
            Tuple::vector(19.86, -30.48, -39.13),
            Tuple::new(-36.32, -49.83, 13.11, 15.72),
        ];

        examples.iter().all(check)
    }

    #[quickcheck]
    fn rotation_x() -> bool {
        rotation(Matrix::rotation_x, |u| Tuple::new(u.x, -u.z, u.y, u.w))
    }

    #[quickcheck]
    fn rotation_y() -> bool {
        rotation(Matrix::rotation_y, |u| Tuple::new(u.z, u.y, -u.x, u.w))
    }

    #[quickcheck]
    fn rotation_z() -> bool {
        rotation(Matrix::rotation_z, |u| Tuple::new(-u.y, u.x, u.z, u.w))
    }

    fn rotation_invert<F: Fn(Float) -> Matrix>(angle: Float, rotation: F) -> bool {
        let mut m = rotation(angle);
        m.invert() && m.eq_approx(&rotation(-angle))
    }

    #[quickcheck]
    fn rotation_invert_x(Finite(angle): Finite) -> bool {
        rotation_invert(angle, Matrix::rotation_x)
    }

    #[quickcheck]
    fn rotation_invert_y(Finite(angle): Finite) -> bool {
        rotation_invert(angle, Matrix::rotation_y)
    }

    #[quickcheck]
    fn rotation_invert_z(Finite(angle): Finite) -> bool {
        rotation_invert(angle, Matrix::rotation_z)
    }

    fn rotate<F: Fn(&mut Matrix, Float), G: Fn(Float) -> Matrix>(
        m: Matrix,
        angle: Float,
        rotate: F,
        rotation: G,
    ) -> bool {
        let mut m2 = m.clone();
        rotate(&mut m2, angle);

        let mut m3 = rotation(angle);
        m3.mul(&m);

        m2.eq_approx(&m3)
    }

    #[quickcheck]
    fn rotate_x(m: Matrix, Finite(angle): Finite) -> bool {
        rotate(m, angle, Matrix::rotate_x, Matrix::rotation_x)
    }

    #[quickcheck]
    fn rotate_y(m: Matrix, Finite(angle): Finite) -> bool {
        rotate(m, angle, Matrix::rotate_y, Matrix::rotation_y)
    }

    #[quickcheck]
    fn rotate_z(m: Matrix, Finite(angle): Finite) -> bool {
        rotate(m, angle, Matrix::rotate_z, Matrix::rotation_z)
    }

    #[quickcheck]
    fn shearing(
        Finite(xy): Finite,
        Finite(xz): Finite,
        Finite(yx): Finite,
        Finite(yz): Finite,
        Finite(zx): Finite,
        Finite(zy): Finite,
        u: Tuple,
    ) -> bool {
        let m = Matrix::shearing(xy, xz, yx, yz, zx, zy);
        (&m * u).eq_approx(Tuple::new(
            u.x + xy * u.y + xz * u.z,
            yx * u.x + u.y + yz * u.z,
            zx * u.x + zy * u.y + u.z,
            u.w,
        ))
    }

    #[quickcheck]
    fn shear(
        m: Matrix,
        Finite(xy): Finite,
        Finite(xz): Finite,
        Finite(yx): Finite,
        Finite(yz): Finite,
        Finite(zx): Finite,
        Finite(zy): Finite,
    ) -> bool {
        let mut m2 = m.clone();
        m2.shear(xy, xz, yx, yz, zx, zy);

        let mut m3 = Matrix::shearing(xy, xz, yx, yz, zx, zy);
        m3.mul(&m);

        m2.eq_approx(&m3)
    }

    #[quickcheck]
    fn transformations_example() -> bool {
        let mut m = Matrix::rotation_x(consts::FRAC_PI_2);
        m.scale(5.0, 5.0, 5.0);
        m.translate(10.0, 5.0, 7.0);
        (&m * Tuple::point(1.0, 0.0, 1.0)).eq_approx(Tuple::point(15.0, 0.0, 7.0))
    }
}
