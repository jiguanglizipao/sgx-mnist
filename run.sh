#!/usr/bin/env bash

mkdir -p data/mnist
curl http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz --output "./data/mnist/train-images-idx3-ubyte.gz"
curl http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz --output "./data/mnist/train-labels-idx1-ubyte.gz"
curl http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz --output "./data/mnist/t10k-images-idx3-ubyte.gz"
curl http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz --output "./data/mnist/t10k-labels-idx1-ubyte.gz"
gzip -d data/mnist/*.gz

cargo run --bin train --release
cargo run --bin encrypt_input --release
cat model.json encrypt_input.json | cargo run --bin predict --target x86_64-fortanix-unknown-sgx --release > encrypt_output.json
cargo run --bin decrypt_output --release
