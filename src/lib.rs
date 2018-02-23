#![cfg_attr(all(feature = "nightly", test), feature(test))]

extern crate ndarray;
extern crate rand;
extern crate num_traits;
#[macro_use] extern crate log;

#[cfg(all(feature = "nightly", test))]
extern crate test;

use num_traits::Float;

use std::ops;
use std::fmt;

pub type Matrix<T> = ndarray::Array2<T>;

pub trait LapJVCost: Float + ops::AddAssign + ops::SubAssign + std::fmt::Debug {}
impl <T>LapJVCost for T where T: Float + ops::AddAssign + ops::SubAssign + std::fmt::Debug {}

#[derive(Debug)]
pub struct LapJVError(&'static str);

impl std::fmt::Display for LapJVError {
    fn fmt(&self, f: &mut fmt::Formatter) -> Result<(), fmt::Error> {
        write!(f, "{}", self.0)
    }
}

impl std::error::Error for LapJVError {
    fn description(&self) -> &str {
        self.0
    }
}

pub struct LapJV<'a, T: 'a> {
    costs: &'a Matrix<T>,
    dim: usize,
    free_rows: Vec<usize>,
    v: Vec<T>,
    in_col: Vec<isize>,
    in_row: Vec<isize>
}

/// Solve LAP problem given cost matrix
/// This is an implementation of the LAPJV algorithm described in:
/// R. Jonker, A. Volgenant. A Shortest Augmenting Path Algorithm for
/// Dense and Sparse Linear Assignment Problems. Computing 38, 325-340
/// (1987)
pub fn lapjv<T>(costs: &Matrix<T>) -> Result<(Vec<isize>, Vec<isize>), LapJVError> where T: LapJVCost {
    LapJV::new(costs).solve()
}

pub fn cost<T>(input: &Matrix<T>, x: &[isize]) -> T where T: LapJVCost {
    (0..x.len()).into_iter().fold(T::zero(), |acc, i| acc + input[(i, x[i] as usize)])
}

/// Solve LAP problem given cost matrix
/// This is an implementation of the LAPJV algorithm described in:
/// R. Jonker, A. Volgenant. A Shortest Augmenting Path Algorithm for
/// Dense and Sparse Linear Assignment Problems. Computing 38, 325-340
/// (1987)
impl <'a, T>LapJV<'a, T> where T: LapJVCost {
    pub fn new(costs: &'a Matrix<T>) -> Self {
        let dim = costs.dim().0; // square matrix dimensions
        let free_rows = Vec::with_capacity(dim); // list of unassigned rows.
        let v = vec![T::max_value(); dim];
        let in_row = vec![-1; dim];
        let in_col = vec![0; dim];
        Self {
            costs,
            dim,
            free_rows,
            v,
            in_col,
            in_row
        }
    }

    pub fn solve(mut self) ->  Result<(Vec<isize>, Vec<isize>), LapJVError> {
        if self.costs.dim().0 != self.costs.dim().1 {
            return Err(LapJVError("Input error: matrix is not square"))
        }
        self.ccrrt_dense();

        let mut i = 0;
        while self.free_rows.len() > 0 && i < 2 {
            self.carr_dense();
            i+= 1;
        }

        if self.free_rows.len() > 0 {
            self.ca_dense()?;
        }

        Ok((self.in_row, self.in_col))
    }

    fn ccrrt_dense(&mut self) {
        let mut unique = vec![true; self.dim];

        for ((i, j), &c) in self.costs.indexed_iter() {
            if c < self.v[j] {
                self.v[j] = c;
                self.in_col[j] = i as isize;
            }
        }

        for j in (0..self.dim).into_iter().rev() {
            let i = self.in_col[j] as usize;
            if self.in_row[i] < 0 {
                self.in_row[i] = j as isize;
            } else {
                unique[i] = false;
                self.in_col[j] = -1;
            }
        }

        for i in 0..self.dim {
            if self.in_row[i] < 0 {
                self.free_rows.push(i);
            } else if unique[i] {
                let j = self.in_row[i];
                let mut min = T::max_value();
                for j2 in 0..self.dim {
                    if j2 == j as usize {
                        continue;
                    }
                    let c = self.costs[(i, j2)] - self.v[j2];
                    if c < min {
                        min = c;
                    }
                }
                self.v[j as usize] -= min;
            }
        }
    }

    // Augmenting row reduction for a dense cost matrix.
    fn carr_dense(&mut self) {
        // AUGMENTING ROW REDUCTION
        // scan all free rows.
        // in some cases, a free row may be replaced with another one to be scanned next.
        let dim = self.dim;
        let mut current = 0;
        let mut new_free_rows = 0; // start list of rows still free after augmenting row reduction.
        let mut rr_cnt = 0;
        let num_free_rows = self.free_rows.len();

        while current < num_free_rows {
            rr_cnt += 1;
            let free_i = self.free_rows[current];
            current += 1;
            // find minimum and second minimum reduced cost over columns.
            let (v1, v2, mut j1, j2) = find_umins_plain(self.costs, free_i, &mut self.v);

            let mut i0 = self.in_col[j1];
            let v1_new = self.v[j1] - (v2 - v1);
            let v1_lowers = v1_new < self.v[j1];  // the trick to eliminate the epsilon bug

            if rr_cnt < current * dim {
                if v1_lowers {
                    // change the reduction of the minimum column to increase the minimum
                    // reduced cost in the row to the subminimum.
                    self.v[j1] = v1_new;
                } else if i0 >= 0 && j2 >= 0 { // minimum and subminimum equal.
                    // minimum column j1 is assigned.
                    // swap columns j1 and j2, as j2 may be unassigned.
                    j1 = j2 as usize;
                    i0 = self.in_col[j2 as usize];
                }
                if i0 >= 0 { // minimum column j1 assigned earlier.
                    if v1_lowers {
                        // put in current k, and go back to that k.
                        // continue augmenting path i - j1 with i0.
                        current -= 1;
                        self.free_rows[current] = i0 as usize;
                    } else {
                        // no further augmenting reduction possible.
                        // store i0 in list of free rows for next phase.
                        self.free_rows[new_free_rows] = i0 as usize;
                        new_free_rows += 1;
                    }
                }
            } else {
                if i0 >= 0 {
                    self.free_rows[new_free_rows] = i0 as usize;
                    new_free_rows += 1;
                }
            }
            self.in_row[free_i] = j1 as isize;
            self.in_col[j1] = free_i as isize;
        }
        self.free_rows.truncate(new_free_rows);
    }


    // Augment for a dense cost matrix.
    fn ca_dense(&mut self) -> Result<(), LapJVError> {
        let dim = self.dim;
        let mut pred = vec![0; dim];

        let free_rows = std::mem::replace(&mut self.free_rows, vec![]);
        for freerow in free_rows {
            trace!("looking at free_i={}", freerow);

            let mut i = std::usize::MAX;
            let mut k = 0;
            let mut j = self.find_path_dense(freerow, &mut pred);
            assert!(j < dim);
            while i != freerow {
                i = pred[j];
                self.in_col[j] = i as isize;
                let tmp = j;
                j = self.in_row[i] as usize;
                self.in_row[i] = tmp as isize;
                k += 1;
                if k >= dim {
                    return Err(LapJVError("Error: ca_dense will not finish"))
                }
            }
        }
        Ok(())
    }

        /// Single iteration of modified Dijkstra shortest path algorithm as explained in the JV paper.
    /// return The closest free column index.
    fn find_path_dense(&mut self, start_i: usize, pred: &mut [usize]) -> usize {
        let dim = self.dim;
        let mut collist = vec![0; dim]; // list of columns to be scanned in various ways.
        let mut d = vec![T::zero(); dim]; // 'cost-distance' in augmenting path calculation.

        let mut lo = 0;
        let mut hi = 0;
        let mut n_ready = 0;

        // Dijkstra shortest path algorithm.
        // runs until unassigned column added to shortest path tree.
        for i in 0..dim {
            collist[i] = i;
            pred[i] = start_i;
            d[i] = self.costs[(start_i, i)] - self.v[i];
        }

        trace!("d: {:?}", d);
        let mut final_j = -1isize;
        while  final_j == -1 {
            if lo == hi {
                trace!("{}..{} -> find", lo, hi);
                n_ready = lo;
                hi = find_dense(dim, lo, &d, &mut collist);
                trace!("check {}..{}", lo, hi);
                // check if any of the minimum columns happens to be unassigned.
                // if so, we have an augmenting path right away.
                for &j in collist.iter().take(hi).skip(lo) {
                    if self.in_col[j] < 0 {
                        final_j = j as isize;
                    }
                }
            }

            if final_j == -1 {
                trace!("{}..{} -> scan", lo, hi);
                let maybe_final_j = self.scan_dense(&mut lo, &mut hi, &mut d, &mut collist, pred);
                if let Some(val) = maybe_final_j {
                    final_j = val as isize;
                }
            }
        }

        trace!("found final_j={}", final_j);
        let mind = d[collist[lo]];
        for &j in collist.iter().take(n_ready) {
            self.v[j] += d[j] - mind;
        }
        final_j as usize
    }
    // Scan all columns in TODO starting from arbitrary column in SCAN
    // and try to decrease d of the TODO columns using the SCAN column.
    fn scan_dense(&self, plo: &mut usize, phi: &mut usize, d: &mut [T], collist: &mut [usize], pred: &mut [usize]) -> Option<usize>  {
        let mut lo = *plo;
        let mut hi = *phi;
        while lo != hi {
            let j = collist[lo];
            lo += 1;
            let i = self.in_col[j] as usize;
            let mind = d[j];
            let h = self.costs[(i, j)] - self.v[j] - mind;
            // For all columns in TODO
            for k in hi..collist.len() {
                let j = collist[k];
                let cred_ij = self.costs[(i, j)] - self.v[j] - h;
                if cred_ij < d[j] {
                    d[j] = cred_ij;
                    pred[j] = i;
                    if (cred_ij - mind).abs() < T::epsilon() {
                    // if cred_ij == mind {
                        if self.in_col[j] < 0 {
                            return Some(j);
                        }
                        collist[k] = collist[hi];
                        collist[hi] = j;
                        hi += 1;
                    }
                }
            }
        }
        // Note: only change lo and hi if the item was not found
        *plo = lo;
        *phi = hi;
        None
    }
}

fn find_dense<T>(dim: usize, lo: usize, d: &[T], collist: &mut [usize]) -> usize  where T: LapJVCost {
    let mut hi  = lo + 1;
    let mut mind = d[collist[lo]];
    for k in hi..dim {
        let j = collist[k];
        let h = d[j];
        if h <= mind {
            if h < mind { // new minimum.
                hi = lo; // restart list at index low.
                mind = h;
            }
            // new index with same minimum, put on undex up, and extend list.
            collist[k] = collist[hi];
            collist[hi] = j;
            hi += 1;
        }
    }
    hi
}

// Finds minimum and second minimum from a row, returns (min, second_min, min_index, second_min_index)
#[inline(always)]
fn find_umins_plain<T>(matrix: &Matrix<T>, row: usize, v: &[T]) -> (T, T, usize, isize)  where T: LapJVCost {
    let local_cost = matrix.row(row);
    let mut umin = local_cost[0] - v[0];
    let mut usubmin = T::max_value();
    let mut j1 = 0;
    let mut j2 = -1;
    for j in 1..local_cost.dim() {
        let h = local_cost[j] - v[j as usize];
        if h < usubmin {
            if h >= umin {
                usubmin = h;
                j2 = j as isize;
            } else {
                usubmin = umin;
                umin = h;
                j2 = j1 as isize;
                j1 = j;
            }
        }
    }
    (umin, usubmin, j1, j2)
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand;

    #[test]
    fn it_works() {
        let m = Matrix::from_shape_vec((3,3), vec![1.0,2.0,3.0,4.0,5.0,6.0,7.0,8.0,9.0]).unwrap();
        let result = lapjv(&m).unwrap();
        assert_eq!(result.0, vec![2, 0, 1]);
        assert_eq!(result.1, vec![1, 2, 0]);
    }

    #[test]
    fn solve_matrix_txt() {
        use std::io::Read;
        let mut src = String::new();
        std::fs::File::open("tests/matrix.csv").unwrap().read_to_string(&mut src).unwrap();
        let vec = src.split('\n').flat_map(|line| line.split_terminator(',').map(|num| num.parse::<f64>().unwrap() )).collect();
        let m = Matrix::from_shape_vec((2366,2366), vec).unwrap();
        let result = lapjv(&m).unwrap();
        let cost = cost(&m, &result.0);
        println!("cost: {}", cost);
        assert!((cost - 30493.6755).abs() < 1e-3);
    }

    #[test]
    fn test_solve_random10() {
        let (m, result) = solve_random10();
        let cost = cost(&m, &result.0);
        assert_eq!(cost, 1071.0);
        assert_eq!(result.0, vec![7,9,3,4,1,0,5,6,2,8]);
    }

    #[test]
    fn test_solve_inf1() {
        let c = vec![
            std::f64::INFINITY, 643.0, 717.0,   2.0, 946.0, 534.0, 242.0, 235.0, 376.0, 839.0,
            std::f64::INFINITY, 141.0, 799.0, 180.0, 386.0, 745.0, 592.0, 822.0, 421.0,  42.0,
            std::f64::INFINITY, 369.0, 831.0,  67.0, 258.0, 549.0, 615.0, 529.0, 458.0, 524.0,
            std::f64::INFINITY, 649.0, 287.0, 910.0,  12.0, 820.0,  31.0,  92.0, 217.0, 555.0,
            std::f64::INFINITY,  81.0, 568.0, 241.0, 292.0, 653.0, 417.0, 652.0, 630.0, 788.0,
            std::f64::INFINITY, 822.0, 788.0, 166.0, 122.0, 690.0, 304.0, 568.0, 449.0, 214.0,
            std::f64::INFINITY, 469.0, 584.0, 633.0, 213.0, 414.0, 498.0, 500.0, 317.0, 391.0,
            std::f64::INFINITY, 581.0, 183.0, 420.0,  16.0, 748.0,  35.0, 516.0, 639.0, 356.0,
            std::f64::INFINITY, 921.0,  67.0,  33.0, 592.0, 775.0, 780.0, 335.0, 464.0, 788.0,
            std::f64::INFINITY, 455.0, 950.0,  25.0,  22.0, 576.0, 969.0, 122.0,  86.0,  74.0,
        ];
        let m = Matrix::from_shape_vec((10,10), c).unwrap();
        let result = lapjv(&m).unwrap();
        assert_eq!(result.0, vec![7, 9, 3, 0, 1, 4, 5, 6, 2, 8]);
    }


    #[test]
    fn test_find_umins() {
        let m = Matrix::from_shape_vec((3,3), vec![25.0,0.0,15.0,4.0,5.0,6.0,7.0,8.0,9.0]).unwrap();
        let result = find_umins_plain(&m, 0, &vec![0.0,0.0,0.0]);
        println!("Result: {:?}", result);
        assert_eq!(result,(0.0, 15.0, 1, 2));
    }

    #[test]
    fn test_random() {
        use rand::Rng;
        let mut rng = rand::thread_rng();

        const DIM: usize = 512;
        let mut m = Vec::with_capacity(DIM*DIM);
        for _ in 0..DIM*DIM {
            m.push(rng.next_f64()*100.0);
        }
        let m = Matrix::from_shape_vec((DIM,DIM), m).unwrap();
        let _result = lapjv(&m).unwrap();
    }

    fn solve_random10() -> (Matrix<f64>, (Vec<isize>, Vec<isize>)) {
        const N: usize = 10;
        let c = vec![
            612.0, 643.0, 717.0,   2.0, 946.0, 534.0, 242.0, 235.0, 376.0, 839.0,
            224.0, 141.0, 799.0, 180.0, 386.0, 745.0, 592.0, 822.0, 421.0,  42.0,
            241.0, 369.0, 831.0,  67.0, 258.0, 549.0, 615.0, 529.0, 458.0, 524.0,
            231.0, 649.0, 287.0, 910.0,  12.0, 820.0,  31.0,  92.0, 217.0, 555.0,
            912.0,  81.0, 568.0, 241.0, 292.0, 653.0, 417.0, 652.0, 630.0, 788.0,
            32.0, 822.0, 788.0, 166.0, 122.0, 690.0, 304.0, 568.0, 449.0, 214.0,
            441.0, 469.0, 584.0, 633.0, 213.0, 414.0, 498.0, 500.0, 317.0, 391.0,
            798.0, 581.0, 183.0, 420.0,  16.0, 748.0,  35.0, 516.0, 639.0, 356.0,
            351.0, 921.0,  67.0,  33.0, 592.0, 775.0, 780.0, 335.0, 464.0, 788.0,
            771.0, 455.0, 950.0,  25.0,  22.0, 576.0, 969.0, 122.0,  86.0,  74.0,
        ];
        let m = Matrix::from_shape_vec((N,N), c).unwrap();
        let result = lapjv(&m).unwrap();
        (m, result)
    }

    #[cfg(feature = "nightly")]
    mod benches {
        use test::Bencher;
        use super::*;

        #[bench]
        fn bench_solve_random10(b: &mut Bencher) {
            b.iter(|| test_solve_random10());
        }

        #[bench]
        fn bench_solve_random_inf1(b: &mut Bencher) {
            b.iter(|| test_solve_random10());
        }
    }
}
