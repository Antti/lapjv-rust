# lapjv

[![Crates.io](https://img.shields.io/crates/v/lapjv.svg)](https://crates.io/crates/lapjv) [![Crates.io](https://img.shields.io/crates/d/lapjv.svg)](https://crates.io/crates/lapjv)
[![Build Status](https://travis-ci.org/Antti/lapjv-rust.svg?branch=master)](https://travis-ci.org/Antti/lapjv-rust)

## Linear Assignment Problem solver using Jonker-Volgenant algorithm


This is rust implementation of the Jonker-Volgenant algorithm for linear assignment problem

* [documentation](https://docs.rs/lapjv/)
* [website](https://github.com/Antti/lapjv/)

## Example usage:

```rust
use lapjv::lapjv;

let m = Matrix::from_shape_vec((3, 3), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]).unwrap();
let result = lapjv(&m).unwrap();
assert_eq!(result.0, vec![2, 0, 1]);
assert_eq!(result.1, vec![1, 2, 0]);
```
