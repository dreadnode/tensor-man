FROM rust:bullseye as builder

RUN apt-get update && apt-get install -y libssl-dev ca-certificates cmake git

WORKDIR /app
ADD . /app
RUN cargo build --release

FROM debian:bullseye

RUN apt-get update && apt-get install -y libssl-dev curl ca-certificates

RUN curl -fsSL https://get.docker.com | sh

COPY --from=builder /app/target/release/tman /usr/bin/tman

ENTRYPOINT ["/usr/bin/tman"]