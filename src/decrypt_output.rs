extern crate autograd as ag;
#[macro_use(s)]
extern crate ndarray;
extern crate serde;
extern crate serde_json;
extern crate bincode;

#[macro_use] extern crate hex_literal;
extern crate aes;
extern crate block_modes;

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

    let y_test = dataset::load();
    let y_slice = y_test.slice(s![0..test_num, ..]);

    let path = Path::new("encrypt_output.json");
    let display = path.display();

    let mut file = match File::open(&path) {
        Err(why) => panic!("couldn't open {}: {}", display, why.description()),
        Ok(file) => file,
    };

    let mut serialize_ciphertext = String::new();
    match file.read_to_string(&mut serialize_ciphertext) {
        Err(why) => panic!("couldn't read {}: {}", display, why.description()),
        Ok(_) => print!("load {} successfully\n", display),
    }

    type Aes128Cbc = Cbc<Aes128, Pkcs7>;
    let cipher = Aes128Cbc::new_var(&key, &iv).unwrap();
    let mut cipher_y_pred:Vec<u8> = serde_json::from_str(&serialize_ciphertext).unwrap();
    let serialized_y_pred = cipher.decrypt(cipher_y_pred.as_mut_slice()).unwrap();
    let y_pred:ndarray::Array<f32, ndarray::IxDyn> = bincode::deserialize(&serialized_y_pred).unwrap();

    let y_compare = ndarray::stack![ndarray::Axis(1), y_slice, y_pred];
    println!("{}", y_compare);
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
        let (test_y, num_label_test): (Vec<f32>, usize) =
            load_labels("data/mnist/t10k-labels-idx1-ubyte");

        // Vec to ndarray
        let as_arr = NdArray::from_shape_vec;
        let y_test = as_arr(ndarray::IxDyn(&[num_label_test, 1]), test_y).unwrap();
        y_test
    }

    fn load_labels<P: AsRef<Path>>(path: P) -> (Vec<f32>, usize) {
        let ref mut buf_reader = io::BufReader::new(File::open(path).unwrap());
        let magic = u32::from_be(read_u32(buf_reader));
        if magic != 2049 {
            panic!("Invalid magic number. Got expect 2049, got {}", magic);
        }
        let num_label = u32::from_be(read_u32(buf_reader)) as usize;
        // read labels
        let mut buf: Vec<u8> = vec![0u8; num_label];
        let _ = buf_reader.read_exact(buf.as_mut());
        let ret: Vec<f32> = buf.into_iter().map(|x| x as f32).collect();
        (ret, num_label)
    }

    fn read_u32<T: Read>(reader: &mut T) -> u32 {
        let mut buf: [u8; 4] = [0, 0, 0, 0];
        let _ = reader.read_exact(&mut buf);
        unsafe { mem::transmute(buf) }
    }
}
