use crate::{Float, Tuple};
use std::ops;

#[cfg(test)]
use crate::{eq_approx_eps, EPSILON};

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

    pub fn inverse(&mut self) -> bool {
        let det = self.determinant();

        if det.abs() < 1e-3 {
            return false;
        }

        let mut m = [[0.0; 4]; 4];

        for i in 0..4 {
            for j in 0..4 {
                m[j][i] = self.cofactor(i, j) / det;
            }
        }

        self.0 = m;
        true
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

    #[cfg(test)]
    fn eq_approx(&self, rhs: &Matrix) -> bool {
        self.eq_approx_eps(rhs, EPSILON)
    }

    #[cfg(test)]
    fn eq_approx_eps(&self, rhs: &Matrix, epsilon: Float) -> bool {
        (0..4).all(|i| (0..4).all(|j| eq_approx_eps(self[i][j], rhs[i][j], epsilon)))
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
    use crate::tuple::Tuple;
    use crate::{eq_approx, eq_approx_eps, Float};
    use quickcheck::TestResult;
    use quickcheck::{Arbitrary, Gen};

    impl Arbitrary for Matrix {
        fn arbitrary(gen: &mut Gen) -> Matrix {
            let mut xs = [[0.0; 4]; 4];
            for i in 0..4 {
                for j in 0..4 {
                    let Finite(x) = Arbitrary::arbitrary(gen);
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

    #[derive(Clone, Debug)]
    struct StableMatrix(Matrix);

    impl Arbitrary for StableMatrix {
        fn arbitrary(gen: &mut Gen) -> StableMatrix {
            let mut gen_small_float = || loop {
                let x: Float = Arbitrary::arbitrary(gen);
                if x.abs() < 100.0 {
                    return x;
                }
            };

            loop {
                let mut xs = [[0.0; 4]; 4];
                for i in 0..4 {
                    for j in 0..4 {
                        xs[i][j] = gen_small_float();
                    }
                }
                let m = Matrix::new(xs);
                let det = m.determinant().abs();
                if det < 1e3 {
                    return StableMatrix(m);
                }
            }
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
    fn transpose_identity() -> bool {
        let mut m = Matrix::identity();
        m.transpose();
        m.eq_approx(&Matrix::identity())
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
    fn mul_tuple_distributive(m: Matrix, u: Tuple, v: Tuple) -> bool {
        (&m * (u + v)).eq_approx_eps((&m * u) + (&m * v), 1e-2)
    }

    #[quickcheck]
    fn mul_tuple_linear(m: Matrix, u: Tuple, Finite(x): Finite) -> bool {
        (&m * (u * x)).eq_approx_eps((&m * u) * x, 1e-2)
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
    fn determinant_transpose(StableMatrix(m): StableMatrix) -> bool {
        let mut m2 = m.clone();
        m2.transpose();
        println!("{} {}", m.determinant(), m2.determinant());
        eq_approx_eps(m.determinant(), m2.determinant(), 1e-2)
    }

    #[quickcheck]
    fn inverse(StableMatrix(m): StableMatrix) -> bool {
        let mut m2 = m.clone();

        if m2.inverse() {
            let mut m3 = m.clone();
            m3.mul(&m2);
            m2.mul(&m);
            println!("{:?} {:?}", m2, m3);
            m2.eq_approx_eps(&Matrix::identity(), 0.1) && m3.eq_approx_eps(&Matrix::identity(), 0.1)
        } else {
            m.eq_approx(&m2) && eq_approx(m.determinant(), 0.0)
        }
    }

    #[quickcheck]
    fn inverse_examples() -> bool {
        let mut m = Matrix::new([
            [-5.0, 2.0, 6.0, -8.0],
            [1.0, -5.0, 1.0, 8.0],
            [7.0, 7.0, -6.0, -7.0],
            [1.0, -3.0, 7.0, 4.0],
        ]);
        m.inverse();

        let m2 = Matrix::new([
            [0.21804511, 0.45112782, 0.2406015, -0.04511278],
            [-0.80827068, -1.45676692, -0.44360902, 0.52067669],
            [-0.07894737, -0.22368421, -0.05263158, 0.19736842],
            [-0.52255639, -0.81390977, -0.30075188, 0.30639098],
        ]);

        let mut m3 = Matrix::new([
            [8.0, -5.0, 9.0, 2.0],
            [7.0, 5.0, 6.0, 1.0],
            [-6.0, 0.0, 9.0, 6.0],
            [-3.0, 0.0, -9.0, -4.0],
        ]);
        m3.inverse();

        let m4 = Matrix::new([
            [-0.15384615, -0.15384615, -0.28205128, -0.53846154],
            [-0.07692308, 0.12307692, 0.02564103, 0.03076923],
            [0.35897436, 0.35897436, 0.43589744, 0.92307692],
            [-0.69230769, -0.69230769, -0.76923077, -1.92307692],
        ]);

        let mut m5 = Matrix::new([
            [9.0, 3.0, 0.0, 9.0],
            [-5.0, -2.0, -6.0, -3.0],
            [-4.0, 9.0, 6.0, 4.0],
            [-7.0, 6.0, 6.0, 2.0],
        ]);
        m5.inverse();

        let m6 = Matrix::new([
            [-0.04074074, -0.07777778, 0.14444444, -0.22222222],
            [-0.07777778, 0.03333333, 0.36666667, -0.33333333],
            [-0.02901235, -0.1462963, -0.10925926, 0.12962963],
            [0.17777778, 0.06666667, -0.26666667, 0.33333333],
        ]);

        let mut m7 = Matrix::identity();
        m7.inverse();

        m.eq_approx(&m2)
            && m3.eq_approx(&m4)
            && m5.eq_approx(&m6)
            && m7.eq_approx(&Matrix::identity())
    }

    #[quickcheck]
    fn inverse_transpose(StableMatrix(m): StableMatrix) -> bool {
        let mut m2 = m.clone();
        m2.transpose();
        let m2_inverted = m2.inverse();

        let mut m3 = m.clone();
        let m3_inverted = m3.inverse();
        m3.transpose();

        m2_inverted == m3_inverted && m2.eq_approx_eps(&m3, 0.1)
    }

    #[quickcheck]
    fn example() -> bool {
        let m = Matrix::new([
            [-34.38463, -0.0, 7.0404973, -87.125336],
            [-0.0, -8.659339, -43.700603, -3.41601],
            [1.8594441, 46.295227, -0.0, -0.0],
            [0.0, 78.713684, -13.892131, -9.44291],
        ]);

        let mut m2 = m.clone();
        m2.transpose();
        let m2_inverted = m2.inverse();

        let mut m3 = m.clone();
        let m3_inverted = m3.inverse();
        m3.transpose();

        println!("{:?} {:?}", m2, m3);

        m2_inverted == m3_inverted && m2.eq_approx_eps(&m3, 0.1)
    }
}
