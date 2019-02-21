#[cfg(test)]
extern crate rustc_version;

#[cfg(test)]
use rustc_version::{version_meta, Channel};

#[cfg(test)]
fn main() {
    if version_meta().unwrap().channel == Channel::Nightly {
        println!("cargo:rustc-cfg=feature=\"nightly\"");
    }
}

fn main() {
    
}
