#![feature(test)]

extern crate ndarray;
#[cfg(test)] extern crate test;

pub type Matrix<T> = ndarray::Array2<T>;

/// @brief Jonker-Volgenant algorithm.
/// @param dim in problem size
/// @param assign_cost in cost matrix
/// @param verbose in indicates whether to report the progress to stdout
/// @param rowsol out column assigned to row in solution / size dim
/// @param colsol out row assigned to column in solution / size dim
/// @param u out dual variables, row reduction numbers / size dim
/// @param v out dual variables, column reduction numbers / size dim
/// @return achieved minimum assignment cost

pub fn lapjv(matrix: &Matrix<f64>) -> (Vec<isize>, Vec<isize>) {
    {
        use std::io::Write;
        let mut f = std::fs::File::create("matrix.txt").unwrap();
        f.write_all(format!("{:?}", matrix).as_bytes()).unwrap();
        f.flush().unwrap();
    }
    let dim = matrix.dim().0;
    let mut free = vec![0; dim]; // list of unassigned rows.
    let mut collist = vec![0; dim]; // list of columns to be scanned in various ways.
    let mut matches = vec![0; dim]; // counts how many times a row could be assigned.
    let mut d = vec![0f64; dim]; // 'cost-distance' in augmenting path calculation. // cost
    let mut pred = vec![0; dim]; // row-predecessor of column in augmenting/alternating path.

    let mut v = vec![0f64; dim];

    let mut in_row = vec![0; dim];
    let mut in_col = vec![0; dim];

    // COLUMN REDUCTION
    for j in (0..dim).into_iter().rev() {   // reverse order gives better results.
        let mut min = matrix[(0, j)];
        let mut imin = 0;
        for i in 1..dim {
            if matrix[(i, j)] < min {
                min = matrix[(i,j)];
                imin = i;
            }
        }

		v[j] = min;
		matches[imin] += 1;

        if matches[imin] == 1 {
            // init assignment if minimum row assigned for first time.
            in_row[imin] = j as isize;
            in_col[j] = imin as isize;
        } else {
            in_col[j] = -1; // row already assigned, column not assigned.
        }
    }
    println!("lapjv: column reduction finished");

    // REDUCTION TRANSFER
    let mut numfree = 0;
    for i in 0..dim {
        if matches[i] == 0 {  // fill list of unassigned 'free' rows.
            free[numfree] = i;
            numfree +=1;
        } else if matches[i] == 1 { // transfer reduction from rows that are assigned once.
            let j1 = in_row[i] as usize;
            let mut min = std::f64::MAX;
            for j in 0..dim {
                if j != j1 && matrix[(i,j)] - v[j] < min {
                    min = matrix[(i,j)] - v[j];
                }
            }
            v[j1] -= min;
        }
    }

    println!("lapjv: REDUCTION TRANSFER finished");

    // AUGMENTING ROW REDUCTION
    for loopcmt in 0..2 { // loop to be done twice.
        // scan all free rows.
        // in some cases, a free row may be replaced with another one to be scanned next.
        let mut k = 0;
        let prvnumfree = numfree;
        numfree = 0; // start list of rows still free after augmenting row reduction.

        while k < prvnumfree {
            let i = free[k];
            k += 1;
            // find minimum and second minimum reduced cost over columns.
            let (umin, usubmin, mut j1, mut j2) = find_umins_plain(matrix, i, &v);

            let mut i0 = in_col[j1];
            let vj1_new = v[j1] - (usubmin - umin);
            let vj1_lowers = vj1_new < v[j1];  // the trick to eliminate the epsilon bug

            if vj1_lowers {
                // change the reduction of the minimum column to increase the minimum
                // reduced cost in the row to the subminimum.
                v[j1] = vj1_new;
            } else if i0 >= 0 { // minimum and subminimum equal.
                // minimum column j1 is assigned.
                // swap columns j1 and j2, as j2 may be unassigned.
                j1 = j2;
                i0 = in_col[j2];
            }

            // (re-)assign i to j1, possibly de-assigning an i0.
            in_row[i] = j1 as isize;
            in_col[j1] = i as isize;

            if i0 >= 0 { // minimum column j1 assigned earlier.
                if vj1_lowers {
                    // put in current k, and go back to that k.
                    // continue augmenting path i - j1 with i0.
                    k -= 1;
                    free[k] = i0 as usize;
                } else {
                    // no further augmenting reduction possible.
                    // store i0 in list of free rows for next phase.
                    free[numfree] = i0 as usize;
                    numfree += 1;
                }
            }
        }
        println!("lapjv: augmenting row reduction: {}/{}", loopcmt, 2);
    }

    for f in 0..numfree {
        let freerow = free[f];
        let mut endofpath = 0;
        for j in 0..dim {
            d[j] = matrix[(freerow, j)] - v[j];
            pred[j] = freerow;
            collist[j] = j;
        }

        let mut low = 0;
        let mut up = 0;
        let mut unassignedfound = false;

        while !unassignedfound {
            let mut min = 0f64;
            let mut last = 0;
            if up == low {
                last = low - 1;
                min = d[collist[up]];
                up += 1;

                for k in up..dim {
                    let j = collist[k];
                    let h = d[j];
                    if h <= min {
                        if h < min {
                            up = low;
                            min = h;
                        }
                        collist[k] = collist[up];
                        collist[up] = j;
                        up += 1;
                    }
                }

                for k in low..up {
                    if in_col[collist[k]] < 0 {
                        endofpath = collist[k];
                        unassignedfound = true;
                        break;
                    }
                }
            }

            if !unassignedfound {
                let j1 = collist[low];
                low += 1;
                let i = in_col[j1] as usize;
                let h = matrix[(i, j1)] - v[j1] - min;

                for k in up..dim {
                    let j = collist[k];
                    let v2 = matrix[(i, j)] - v[j] - h;

                    if v2 < d[j] {
                        pred[j] = i;

                        if (v2 - min).abs() < std::f64::EPSILON {
                            if in_col[j] < 0 {
                                endofpath = j;
                                unassignedfound = true;
                                break;
                            } else {
                                collist[k] = collist[up];
                                collist[up] = j;
                                up += 1;
                            }
                        }

                        d[j] = v2;
                    }
                }
            }

            for k in 0..last {
                let j1 = collist[k];
                v[j1] += d[j1] - min;
            }

            let mut i = freerow + 1;
            while i != freerow {
                i = pred[endofpath];
                in_col[endofpath] = i as isize;
                let j1 = endofpath;
                endofpath = in_row[i] as usize;
                in_row[i] = j1 as isize;
            }
        }
    }
    (in_row, in_col)
}


#[inline(always)]
fn find_umins_plain(matrix: &Matrix<f64>, row: usize, v: &[f64]) -> (f64, f64, usize, usize) {
    let local_cost = matrix.row(row);
    let mut umin = local_cost[0] - v[0];
    let mut j1 = 0isize;
    let mut j2 = -1isize;
    let mut usubmin = std::f64::MAX;
    for j in 1..local_cost.dim() {
        let h = local_cost[j] - v[j];
        if h < usubmin {
            if h >= umin {
                usubmin = h;
                j2 = j as isize;
            } else {
                usubmin = umin;
                umin = h;
                j2 = j1;
                j1 = j as isize;
            }
        }
    }
    (umin, usubmin, j1 as usize, j2 as usize)
}

#[cfg(test)]
mod tests {
    use super::*;
    use test::Bencher;

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

    #[bench]
    fn bench_solve_random10(b: &mut Bencher) {
        b.iter(|| test_solve_random10());
    }

    fn solve_random10() -> (Matrix<f64>, (Vec<isize>, Vec<isize>)) {
        const N: usize = 10;
        let c = vec![
            612, 643, 717,   2, 946, 534, 242, 235, 376, 839,
            224, 141, 799, 180, 386, 745, 592, 822, 421,  42,
            241, 369, 831,  67, 258, 549, 615, 529, 458, 524,
            231, 649, 287, 910,  12, 820,  31,  92, 217, 555,
            912,  81, 568, 241, 292, 653, 417, 652, 630, 788,
            32, 822, 788, 166, 122, 690, 304, 568, 449, 214,
            441, 469, 584, 633, 213, 414, 498, 500, 317, 391,
            798, 581, 183, 420,  16, 748,  35, 516, 639, 356,
            351, 921,  67,  33, 592, 775, 780, 335, 464, 788,
            771, 455, 950,  25,  22, 576, 969, 122,  86,  74,
        ].iter().map(|x| *x as f64).collect();
        let m = Matrix::from_shape_vec((N,N), c).unwrap();
        let result = lapjv(&m);
        (m, result)
    }

    fn cost(input: &Matrix<f64>, (rows, _cols): (&[isize], &[isize])) -> f64 {
        (0..rows.len()).into_iter().fold(0.0, |acc, i| acc + input[(i, rows[i] as usize)])
    }
}
