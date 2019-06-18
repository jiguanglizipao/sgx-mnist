#!/usr/bin/env bash
cargo run --bin train --release
cargo run --bin encrypt_input --release
cat model.json encrypt_input.json | cargo run --bin predict --target x86_64-fortanix-unknown-sgx --release > encrypt_output.json
cargo run --bin decrypt_output --release
