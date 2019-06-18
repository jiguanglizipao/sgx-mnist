extern crate autograd as ag;
#[macro_use(s)]
extern crate ndarray;
extern crate serde;
extern crate serde_json;
extern crate bincode;

#[macro_use] extern crate hex_literal;
extern crate aes;
extern crate block_modes;

use std::time::Instant;
use std::error::Error;
use std::fs::File;
use std::io::prelude::*;
use std::path::Path;

use block_modes::{BlockMode, Cbc};
use block_modes::block_padding::Pkcs7;
use aes::Aes128;

fn main() {
    let key = hex!("000102030405060708090a0b0c0d0e0f");
    let iv = hex!("f0f1f2f3f4f5f6f7f8f9fafbfcfdfeff");
    let test_num = 20;

    let x_test = dataset::load();
    let x_slice = x_test.slice(s![0..test_num, .., .., ..]);
    let serialized_x_test = bincode::serialize(&x_slice).unwrap();

    type Aes128Cbc = Cbc<Aes128, Pkcs7>;
    let cipher = Aes128Cbc::new_var(&key, &iv).unwrap();
    let ciphertext = cipher.encrypt_vec(serialized_x_test.as_slice());

    let serialize_ciphertext = serde_json::to_string(&ciphertext).unwrap()+"\n";

    let path = Path::new("encrypt_input.json");
    let display = path.display();

    let mut file = match File::create(&path) {
        Err(why) => panic!("couldn't create {}: {}", display, why.description()),
        Ok(file) => file,
    };

    match file.write_all(serialize_ciphertext.as_bytes()) {
        Err(why) => panic!("couldn't write to {}: {}", display, why.description()),
        Ok(_) => println!("successfully wrote to {}", display),
    }

}

pub mod dataset {
    extern crate ndarray;
    use std::fs::File;
    use std::io;
    use std::io::Read;
    use std::mem;
    use std::path::Path;

    type NdArray = ndarray::Array<f32, ndarray::IxDyn>;

    /// load mnist dataset as "ndarray" objects.
    ///
    /// labels are sparse (vertical vector).
    pub fn load() -> NdArray {
        // load dataset as `Vec`s
        let (test_x, num_image_test): (Vec<f32>, usize) =
            load_images("data/mnist/t10k-images-idx3-ubyte");

        // Vec to ndarray
        let as_arr = NdArray::from_shape_vec;
        let x_test = as_arr(ndarray::IxDyn(&[num_image_test, 1, 28, 28]), test_x).unwrap();
        x_test
    }

    fn load_images<P: AsRef<Path>>(path: P) -> (Vec<f32>, usize) {
        let ref mut buf_reader = io::BufReader::new(
            File::open(path).expect("Please run ./download_mnist.sh beforehand"),
        );
        let magic = u32::from_be(read_u32(buf_reader));
        if magic != 2051 {
            panic!("Invalid magic number. expected 2051, got {}", magic)
        }
        let num_image = u32::from_be(read_u32(buf_reader)) as usize;
        let rows = u32::from_be(read_u32(buf_reader)) as usize;
        let cols = u32::from_be(read_u32(buf_reader)) as usize;
        assert!(rows == 28 && cols == 28);

        // read images
        let mut buf: Vec<u8> = vec![0u8; num_image * rows * cols];
        let _ = buf_reader.read_exact(buf.as_mut());
        let ret = buf.into_iter().map(|x| (x as f32) / 255.).collect();
        (ret, num_image)
    }

    fn read_u32<T: Read>(reader: &mut T) -> u32 {
        let mut buf: [u8; 4] = [0, 0, 0, 0];
        let _ = reader.read_exact(&mut buf);
        unsafe { mem::transmute(buf) }
    }
}
