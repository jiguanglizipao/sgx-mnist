extern crate autograd as ag;
#[macro_use(s)]
extern crate ndarray;
extern crate serde;
extern crate serde_json;

use std::time::Instant;
use std::error::Error;
use std::fs::File;
use std::io::prelude::*;
use std::path::Path;

type Tensor = ag::Tensor<f32>;

// This is a toy convolutional network for MNIST.
// Got 0.985 test accuracy in 300 sec on 2.7GHz Intel Core i5.
//
// First, run "./download_mnist.sh" beforehand if you don't have dataset and then run
// "cargo run --example cnn_mnist --release" in `examples` directory.
macro_rules! timeit {
    ($x:expr) => {{
        let start = Instant::now();
        let result = $x;
        let end = start.elapsed();
        println!(
            "{}.{:03} sec",
            end.as_secs(),
            end.subsec_nanos() / 1_000_000
        );
        result
    }};
}

fn conv_pool(x: &Tensor, w: &Tensor, b: &Tensor) -> Tensor {
    let y1 = ag::conv2d(x, w, 1, 1) + b;
    let y2 = ag::relu(y1);
    ag::max_pool2d(y2, 2, 0, 2)
}

fn logits(x: &Tensor, w: &Tensor, b: &Tensor) -> Tensor {
    ag::matmul(x, w) + b
}

fn inputs() -> (Tensor, Tensor) {
    let x = ag::placeholder(&[-1, 1, 28, 28]);
    let y = ag::placeholder(&[-1, 1]);
    (x, y)
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
    let ((x_train, y_train), (x_test, y_test)) = dataset::load();

    // -- load --
    let path = Path::new("model.json");
    let display = path.display();

    // Open the path in read-only mode, returns `io::Result<File>`
    let mut file = match File::open(&path) {
        // The `description` method of `io::Error` returns a string that
        // describes the error
        Err(why) => panic!("couldn't open {}: {}", display,
                                                   why.description()),
        Ok(file) => file,
    };

    // Read the file contents into a string, returns `io::Result<usize>`
    let mut parameters_json = String::new();
    match file.read_to_string(&mut parameters_json) {
        Err(why) => panic!("couldn't read {}: {}", display,
                                                   why.description()),
        Ok(_) => print!("load {} successfully\n", display),
    }
    let parameters = Parameters::from_json(&parameters_json);

    let ref w1 = ag::variable(parameters.w1);
    let ref w2 = ag::variable(parameters.w2);
    let ref w3 = ag::variable(parameters.w3);
    let ref b1 = ag::variable(parameters.b1);
    let ref b2 = ag::variable(parameters.b2);
    let ref b3 = ag::variable(parameters.b3);
    let params = &[w1, w2, w3, b1, b2, b3];
    let ref params_adam = ag::gradient_descent_ops::Adam::vars_with_states(params);
    let (x, y) = inputs();
    let z1 = conv_pool(&x, w1, b1); // map to 32 channel
    let z2 = conv_pool(&z1, w2, b2); // map to 64 channel
    let z3 = ag::reshape(z2, &[-1, 64 * 7 * 7]); // flatten
    let logits = logits(&z3, w3, b3); // linear
    let loss = ag::sparse_softmax_cross_entropy(&logits, &y);
    let mean_loss = ag::reduce_mean(loss, &[0, 1], false);
    let grads = &ag::grad(&[&mean_loss], params);
    let adam = ag::gradient_descent_ops::Adam::default();
    let update_ops: &[Tensor] = &adam.compute_updates(params_adam, grads);

    // -- test --
    let predictions = ag::argmax(logits, -1, true);
    let accuracy = ag::reduce_mean(&ag::equal(predictions, &y), &[0, 1], false);
    println!(
        "test accuracy: {:?}",
        accuracy.eval(&[(&x, &x_test), (&y, &y_test)])
    );
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
    pub fn load() -> ((NdArray, NdArray), (NdArray, NdArray)) {
        // load dataset as `Vec`s
        let (train_x, num_image_train): (Vec<f32>, usize) =
            load_images("data/mnist/train-images-idx3-ubyte");
        let (train_y, num_label_train): (Vec<f32>, usize) =
            load_labels("data/mnist/train-labels-idx1-ubyte");
        let (test_x, num_image_test): (Vec<f32>, usize) =
            load_images("data/mnist/t10k-images-idx3-ubyte");
        let (test_y, num_label_test): (Vec<f32>, usize) =
            load_labels("data/mnist/t10k-labels-idx1-ubyte");

        // Vec to ndarray
        let as_arr = NdArray::from_shape_vec;
        let x_train = as_arr(ndarray::IxDyn(&[num_image_train, 1, 28, 28]), train_x).unwrap();
        let y_train = as_arr(ndarray::IxDyn(&[num_label_train, 1]), train_y).unwrap();
        let x_test = as_arr(ndarray::IxDyn(&[num_image_test, 1, 28, 28]), test_x).unwrap();
        let y_test = as_arr(ndarray::IxDyn(&[num_label_test, 1]), test_y).unwrap();
        ((x_train, y_train), (x_test, y_test))
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
