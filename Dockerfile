FROM tozd/sgx:ubuntu-xenial

ENV RUSTUP_HOME=/usr/local/rustup \
    CARGO_HOME=/usr/local/cargo \
    PATH=/usr/local/cargo/bin:$PATH

RUN apt-get update; \
    apt-get install -y --no-install-recommends curl pkg-config libssl-dev protobuf-compiler; \
    curl -s -S -L https://sh.rustup.rs -sSf | sh -s -- -y --no-modify-path --default-toolchain nightly-2019-03-24; \
    chmod -R a+w $RUSTUP_HOME $CARGO_HOME; \
    rustup --version; \
    cargo --version; \
    rustc --version; \
    rustup target add x86_64-fortanix-unknown-sgx; \
    cargo install fortanix-sgx-tools sgxs-tools; \
    apt-get clean && rm -rf /var/lib/apt/lists/* 

WORKDIR /mnist
COPY . .
