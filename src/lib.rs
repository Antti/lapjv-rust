#![cfg_attr(all(feature = "nightly", test), feature(test))]

extern crate ndarray;
extern crate rand;
extern crate num_traits;
#[cfg(all(feature = "nightly", test))]
extern crate test;

use num_traits::{Bounded, Zero};

use std::ops;

pub type Matrix<T> = ndarray::Array2<T>;

pub trait LapJVCost: Bounded + Clone + Copy + PartialOrd + ops::Sub<Output=Self> + ops::AddAssign + ops::SubAssign + Zero {}
impl <T>LapJVCost for T where T: Bounded + Clone + Copy + PartialOrd + ops::Sub<Output=Self> + ops::AddAssign + ops::SubAssign + Zero {}

pub struct LapJV<T> {
    matrix: Matrix<T>,
    v: Vec<T>,
    free_rows: Vec<usize>,
    in_row: Vec<isize>,
    in_col: Vec<isize>,
    num_free_rows: usize
}

impl <T>LapJV<T>  where T: LapJVCost {
    pub fn new(matrix: Matrix<T>) -> Self {
        let dim = matrix.dim().0;
        Self {
            matrix,
            free_rows: vec![0; dim],
            v: vec![T::max_value(); dim],
            in_row: vec![-1; dim],
            in_col: vec![0; dim],
            num_free_rows: 0
        }
    }

    pub fn solve(mut self) -> (Vec<isize>, Vec<isize>) {
        let mut num_free_rows = ccrrt_dense(&self.matrix, &mut self.free_rows, &mut self.in_row, &mut self.in_col, &mut self.v);

        let mut i = 0;
        while num_free_rows > 0 && i < 2 {
            num_free_rows = carr_dense(&self.matrix, self.num_free_rows, &mut self.free_rows, &mut self.in_row, &mut self.in_col, &mut self.v);
            i+= 1;
        }
        if num_free_rows > 0 {
            ca_dense(&self.matrix, self.num_free_rows, &mut self.free_rows, &mut self.in_row, &mut self.in_col, &mut self.v);
        }
        (self.in_row, self.in_col)
    }
}

/// Jonker-Volgenant algorithm.
pub fn lapjv<T>(matrix: &Matrix<T>) -> (Vec<isize>, Vec<isize>) where T: LapJVCost {
    let dim = matrix.dim().0; // square matrix dimensions
    let mut free_rows = vec![0; dim]; // list of unassigned rows.

    let mut v = vec![T::max_value(); dim];
    let mut in_row = vec![-1; dim];
    let mut in_col = vec![0; dim];

    let mut num_free_rows = ccrrt_dense(matrix, &mut free_rows, &mut in_row, &mut in_col, &mut v);

    let mut i = 0;
    while num_free_rows > 0 && i < 2 {
        num_free_rows = carr_dense(matrix, num_free_rows, &mut free_rows, &mut in_row, &mut in_col, &mut v);
        i+= 1;
        println!("lapjv: augmenting row reduction: {}/{}", i, 2);
    }
    if num_free_rows > 0 {
        println!("lapjv: ca_dense with num_free_rows: {}", num_free_rows);
        ca_dense(matrix, num_free_rows, &mut free_rows, &mut in_row, &mut in_col, &mut v);
    }
    (in_row, in_col)
}

fn ccrrt_dense<T>(matrix: &Matrix<T>, free_rows: &mut [usize], in_row: &mut [isize], in_col: &mut[isize], v: &mut [T]) -> usize where T: LapJVCost {
    let dim = matrix.dim().0;
    let mut n_free_rows = 0;
    let mut unique = vec![true; dim];

    for i in 0..dim {
        for j in 0..dim {
            let c = matrix[(i,j)];
            if c < v[j] {
                v[j] = c;
                in_col[j] = i as isize;
            }
        }
    }

    for j in (0..dim).into_iter().rev() {
        let i = in_col[j] as usize;
        if in_row[i] < 0 {
            in_row[i] = j as isize;
        } else {
            unique[i] = false;
            in_col[j] = -1;
        }
    }

    for i in 0..dim {
        if in_row[i] < 0 {
            free_rows[n_free_rows] = i;
            n_free_rows +=  1;
        } else if unique[i] {
            let j = in_row[i];
            let mut min = T::max_value();
            for j2 in 0..dim {
                if j2 == j as usize {
                    continue;
                }
                let c = matrix[(i, j2)] - v[j2];
                if c < min {
                    min = c;
                }
            }
            v[j as usize] -= min;
        }
    }
    n_free_rows
}

// Augment for a dense cost matrix.
fn ca_dense<T>(matrix: &Matrix<T>, num_free_rows: usize, free_rows: &mut [usize], in_row: &mut [isize], in_col: &mut [isize], v: &mut [T])  where T: LapJVCost {
    let dim = matrix.dim().0;
    let mut pred = vec![0; dim];

    for f in 0..num_free_rows {
        let freerow = free_rows[f];
        // println!("looking at free_i={}", freerow);

        let mut i = std::usize::MAX;
        let mut k = 0;
        let mut j = find_path_dense(matrix, freerow, in_col, v, &mut pred);
        assert!(j < dim);
        while i != freerow {
            i = pred[j];
            in_col[j] = i as isize;
            let tmp = j;
            j = in_row[i] as usize;
            in_row[i] = tmp as isize;
            k += 1;
            if k >= dim {
                panic!("ca_dense failed");
            }
        }
    }
}

// Augmenting row reduction for a dense cost matrix.
fn carr_dense<T>(matrix: &Matrix<T>, num_free_rows: usize, free_rows: &mut [usize], in_row: &mut [isize], in_col: &mut [isize], v: &mut [T]) -> usize where T: LapJVCost {
    // AUGMENTING ROW REDUCTION
    // scan all free rows.
    // in some cases, a free row may be replaced with another one to be scanned next.
    let dim = matrix.dim().0;
    let mut current = 0;
    let mut new_free_rows = 0; // start list of rows still free after augmenting row reduction.
    let mut rr_cnt = 0;

    // println!("X {:?}", in_row);
    // println!("Y {:?}", in_col);
    // println!("V {:?}", v);
    // println!("F({}) {:?}", num_free_rows, &free_rows[0..num_free_rows]);
    while current < num_free_rows {
        rr_cnt += 1;
        let free_i = free_rows[current];
        current += 1;
        // find minimum and second minimum reduced cost over columns.
        let (v1, v2, mut j1, j2) = find_umins_plain(matrix, free_i, v);

        let mut i0 = in_col[j1];
        let v1_new = v[j1] - (v2 - v1);
        let v1_lowers = v1_new < v[j1];  // the trick to eliminate the epsilon bug

        if rr_cnt < current * dim {
            if v1_lowers {
                // change the reduction of the minimum column to increase the minimum
                // reduced cost in the row to the subminimum.
                v[j1] = v1_new;
            } else if i0 >= 0 && j2 >= 0 { // minimum and subminimum equal.
                // minimum column j1 is assigned.
                // swap columns j1 and j2, as j2 may be unassigned.
                j1 = j2 as usize;
                i0 = in_col[j2 as usize];
            }

            if i0 >= 0 { // minimum column j1 assigned earlier.
                if v1_lowers {
                    // put in current k, and go back to that k.
                    // continue augmenting path i - j1 with i0.
                    current -= 1;
                    free_rows[current] = i0 as usize;
                } else {
                    // no further augmenting reduction possible.
                    // store i0 in list of free rows for next phase.
                    free_rows[new_free_rows] = i0 as usize;
                    new_free_rows += 1;
                }
            }
        } else {
            if i0 >= 0 {
                free_rows[new_free_rows] = i0 as usize;
                new_free_rows += 1;
            }
        }
        in_row[free_i] = j1 as isize;
        in_col[j1] = free_i as isize;
    }
    new_free_rows
}

/// Single iteration of modified Dijkstra shortest path algorithm as explained in the JV paper.
/// return The closest free column index.
fn find_path_dense<T>(matrix: &Matrix<T>, start_i: usize, in_col: &mut [isize], v: &mut [T], pred: &mut [usize]) -> usize where T: LapJVCost {
    let dim = matrix.dim().0;
    let mut collist = vec![0; dim]; // list of columns to be scanned in various ways.
    let mut d = vec![T::zero(); dim]; // 'cost-distance' in augmenting path calculation. // cost

    let mut lo = 0;
    let mut hi = 0;
    let mut n_ready = 0;

    // Dijkstra shortest path algorithm.
    // runs until unassigned column added to shortest path tree.
    for i in 0..dim {
        collist[i] = i;
        pred[i] = start_i;
        d[i] = matrix[(start_i, i)] - v[i];
    }

    // println!("d: {:?}", d);
    let final_j;
    'outer: loop {
        if lo == hi {
            // println!("{}..{} -> find", lo, hi);
            n_ready = lo;
            hi = find_dense(dim, lo, &mut d, &mut collist);
            // println!("check {}..{}", lo, hi);
            // check if any of the minimum columns happens to be unassigned.
            // if so, we have an augmenting path right away.
            for k in lo..hi {
                let j = collist[k];
                if in_col[j] < 0 {
                    final_j = j;
                    break 'outer;
                }
            }
        }

        // println!("{}..{} -> scan", lo, hi);

        let (new_hi, new_lo, maybe_final_j) = scan_dense(matrix, lo, hi, &mut d, &mut collist, pred, in_col, v);
        hi = new_hi;
        lo = new_lo;
        if let Some(val) = maybe_final_j {
            final_j = val;
            break 'outer;
        }
    }

    let mind = d[collist[lo]];
    for k in 0..n_ready {
        let j = collist[k];
        v[j] += d[j] - mind;
    }
    final_j
}

fn find_dense<T>(dim: usize, lo: usize, d: &mut [T], collist: &mut [usize]) -> usize  where T: LapJVCost {
    let mut hi  = lo + 1;
    let mut min = d[collist[lo]];
    for k in hi..dim {
        let j = collist[k];
        let h = d[j];
        if h <= min {
            if h < min { // new minimum.
                hi = lo; // restart list at index low.
                min = h;
            }
            // new index with same minimum, put on undex up, and extend list.
            collist[k] = collist[hi];
            collist[hi] = j;
            hi+=1;
        }
    }
    hi
}

// Scan all columns in TODO starting from arbitrary column in SCAN
// and try to decrease d of the TODO columns using the SCAN column.
fn scan_dense<T>(matrix: &Matrix<T>, mut lo: usize, mut hi: usize, d: &mut [T], collist: &mut [usize], pred: &mut [usize], in_col: &mut [isize], v: &mut [T]) -> (usize, usize, Option<usize>)  where T: LapJVCost {
    let mut h;
    let mut cred_ij;

    while lo != hi {
        let j = collist[lo];
        lo += 1;
        let i = in_col[j] as usize;
        let mind = d[j];
        h = matrix[(i, j)] - v[j] - mind;
        // For all columns in TODO

        for k in hi..matrix.dim().0 {
            let j = collist[k];
            cred_ij = matrix[(i, j)] - v[j] - h;
            if cred_ij < d[j] {
                d[j] = cred_ij;
                pred[j] = i;
                if cred_ij == mind {
                    if in_col[j] < 0 {
                        return (lo, hi, Some(j));
                    }
                    collist[k] = collist[hi];
                    collist[hi] = j;
                    hi += 1;
                }
            }
        }
    }
    (lo, hi, None)
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
        let result = lapjv(&m);
        assert_eq!(result.0, vec![2, 0, 1]);
        assert_eq!(result.1, vec![1, 2, 0]);
    }

    #[test]
    fn test_solve_random10() {
        let (m, result) = solve_random10();
        let cost = cost(&m, (&result.0, &result.1));
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
        let result = lapjv(&m);
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
        let _result = lapjv(&m);
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
        let result = lapjv(&m);
        (m, result)
    }

    fn cost(input: &Matrix<f64>, (rows, _cols): (&[isize], &[isize])) -> f64 {
        (0..rows.len()).into_iter().fold(0.0, |acc, i| acc + input[(i, rows[i] as usize)])
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
