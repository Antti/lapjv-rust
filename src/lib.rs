#![feature(test)]

extern crate ndarray;
#[cfg(test)] extern crate test;

pub type Matrix<T> = ndarray::Array2<T>;

/// Jonker-Volgenant algorithm.
pub fn lapjv(matrix: &Matrix<f64>) -> (Vec<isize>, Vec<isize>) {
    let dim = matrix.dim().0; // square matrix dimensions
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
    let mut num_free_rows = 0;
    for i in 0..dim {
        if matches[i] == 0 {  // fill list of unassigned 'free' rows.
            free[num_free_rows] = i;
            num_free_rows +=1;
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

    println!("lapjv: reduction transfer finished");

    let mut i = 0;
    while num_free_rows > 0 && i < 2 {
        num_free_rows = carr_dense(matrix, num_free_rows, &mut free, &mut in_col, &mut in_row, &mut v);
        i+= 1;
        println!("lapjv: augmenting row reduction: {}/{}", i, 2);
    }

    eprintln!("lapjv: num_free_rows: {}", num_free_rows);
    for f in 0..num_free_rows {
        let freerow = free[f];
        let mut endofpath = 0;

        // Dijkstra shortest path algorithm.
        // runs until unassigned column added to shortest path tree.
        for j in 0..dim {
            d[j] = matrix[(freerow, j)] - v[j];
            pred[j] = freerow;
            collist[j] = j;
        }

        let mut low = 0; // columns in 0..low-1 are ready, now none.
        let mut up = 0;  // columns in low..up-1 are to be scanned for current minimum, now none.
                         // columns in up..dim-1 are to be considered later to find new minimum,
                         // at this stage the list simply contains all columns
        let mut unassignedfound = false;

        // initialized in the first iteration: low == up == 0
        let mut min = 0f64;
        let mut last = 0;
        while !unassignedfound {
            if up == low { // no more columns to be scanned for current minimum.
                last = low - 1;
                 // scan columns for up..dim-1 to find all indices for which new minimum occurs.
                // store these indices between low..up-1 (increasing up).
                min = d[collist[up]];
                up += 1;

                for k in up..dim {
                    let j = collist[k];
                    let h = d[j];
                    if h <= min {
                        if h < min { // new minimum.
                            up = low; // restart list at index low.
                            min = h;
                        }
                        // new index with same minimum, put on undex up, and extend list.
                        collist[k] = collist[up];
                        collist[up] = j;
                        up += 1;
                    }
                }

                // check if any of the minimum columns happens to be unassigned.
                // if so, we have an augmenting path right away.
                for k in low..up {
                    let j = collist[k];
                    if in_col[j] < 0 {
                        endofpath = j;
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

// Augmenting row reduction for a dense cost matrix.
fn carr_dense(matrix: &Matrix<f64>, num_free_rows: usize, free_rows: &mut [usize], in_col: &mut Vec<isize>, in_row: &mut Vec<isize>, v: &mut Vec<f64>) -> usize {
    // AUGMENTING ROW REDUCTION
    // scan all free rows.
    // in some cases, a free row may be replaced with another one to be scanned next.
    let dim = matrix.dim().0;
    let mut current = 0;
    let mut new_free_rows = 0; // start list of rows still free after augmenting row reduction.
    let mut rr_cnt = 0;

    while current < num_free_rows {
        rr_cnt += 1;
        let free_i = free_rows[current];
        current += 1;
        // find minimum and second minimum reduced cost over columns.
        let (v1, v2, mut j1, j2) = find_umins_plain(matrix, free_i, &v);

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



// Finds minimum and second minimum from a row, returns (min, second_min, min_index, second_min_index)
#[inline(always)]
fn find_umins_plain(matrix: &Matrix<f64>, row: usize, v: &[f64]) -> (f64, f64, usize, isize) {
    let local_cost = matrix.row(row);
    let mut umin = local_cost[0] - v[0];
    let mut usubmin = std::f64::INFINITY;
    let mut j1 = 0;
    let mut j2 = -1;
    for j in 1..local_cost.dim() {
        let h = local_cost[j as usize] - v[j as usize];
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

    #[test]
    fn test_find_umins() {
        let m = Matrix::from_shape_vec((3,3), vec![25.0,0.0,15.0,4.0,5.0,6.0,7.0,8.0,9.0]).unwrap();
        let result = find_umins_plain(&m, 0, &vec![0.0,0.0,0.0]);
        println!("Result: {:?}", result);
        assert_eq!(result,(0.0, 15.0, 1, 2));
    }

    #[test]
    fn test_huge_matrix() {
        // let matrix = include!("../matrix.txt");
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
