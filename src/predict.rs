extern crate autograd as ag;
#[macro_use(s)]
extern crate ndarray;
extern crate serde;
extern crate serde_json;
extern crate bincode;

#[macro_use] extern crate hex_literal;
extern crate aes;
extern crate block_modes;

use std::io::{self, BufRead};

use block_modes::{BlockMode, Cbc};
use block_modes::block_padding::Pkcs7;
use aes::Aes128;

type Tensor = ag::Tensor<f32>;

fn conv_pool(x: &Tensor, w: &Tensor, b: &Tensor) -> Tensor {
    let y1 = ag::conv2d(x, w, 1, 1) + b;
    let y2 = ag::relu(y1);
    ag::max_pool2d(y2, 2, 0, 2)
}

fn logits(x: &Tensor, w: &Tensor, b: &Tensor) -> Tensor {
    ag::matmul(x, w) + b
}

fn inputs() -> Tensor {
    let x = ag::placeholder(&[-1, 1, 28, 28]);
    x
}

#[derive(serde::Serialize, serde::Deserialize)]
struct Parameters {
    pub w1: ndarray::Array<f32, ndarray::Dim<ndarray::IxDynImpl>>,
    pub w2: ndarray::Array<f32, ndarray::Dim<ndarray::IxDynImpl>>,
    pub w3: ndarray::Array<f32, ndarray::Dim<ndarray::IxDynImpl>>,
    pub b1: ndarray::Array<f32, ndarray::Dim<ndarray::IxDynImpl>>,
    pub b2: ndarray::Array<f32, ndarray::Dim<ndarray::IxDynImpl>>,
    pub b3: ndarray::Array<f32, ndarray::Dim<ndarray::IxDynImpl>>,
}

impl Parameters {
    fn as_json(&self)-> String
    {
        return serde_json::to_string(&self).unwrap()
    }

    fn from_json(s: &str)-> Self
    {
        return serde_json::from_str(s).unwrap()
    }
}

fn main() {
    let key = hex!("000102030405060708090a0b0c0d0e0f");
    let iv = hex!("f0f1f2f3f4f5f6f7f8f9fafbfcfdfeff");
    let test_num = 1000;

    // -- load --
    let stdin = io::stdin();
    let mut iterator = stdin.lock().lines();
    let line1 = iterator.next().unwrap().unwrap();
    let line2 = iterator.next().unwrap().unwrap();

    let parameters = Parameters::from_json(&line1);

    type Aes128Cbc = Cbc<Aes128, Pkcs7>;
    let cipher = Aes128Cbc::new_var(&key, &iv).unwrap();
    let mut cipher_x_test:Vec<u8> = serde_json::from_str(&line2).unwrap();
    let serialized_x_test = cipher.decrypt(cipher_x_test.as_mut_slice()).unwrap();
    let x_test:ndarray::Array<f32, ndarray::IxDyn> = bincode::deserialize(&serialized_x_test).unwrap();


    let ref w1 = ag::variable(parameters.w1);
    let ref w2 = ag::variable(parameters.w2);
    let ref w3 = ag::variable(parameters.w3);
    let ref b1 = ag::variable(parameters.b1);
    let ref b2 = ag::variable(parameters.b2);
    let ref b3 = ag::variable(parameters.b3);
    let params = &[w1, w2, w3, b1, b2, b3];
    let ref params_adam = ag::gradient_descent_ops::Adam::vars_with_states(params);
    let x = inputs();
    let z1 = conv_pool(&x, w1, b1); // map to 32 channel
    let z2 = conv_pool(&z1, w2, b2); // map to 64 channel
    let z3 = ag::reshape(z2, &[-1, 64 * 7 * 7]); // flatten
    let logits = logits(&z3, w3, b3); // linear

    // -- test --
    let predictions = ag::argmax(logits, -1, true);
    let results = predictions.eval(&[(&x, &x_test)]).unwrap();

    let cipher = Aes128Cbc::new_var(&key, &iv).unwrap();
    let serialized_results = bincode::serialize(&results).unwrap();
    let ciphertext = cipher.encrypt_vec(serialized_results.as_slice());
    let serialized_ciphertext = serde_json::to_string(&ciphertext).unwrap();
    println!("{}", serialized_ciphertext);
}

