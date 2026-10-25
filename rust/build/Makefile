# ==============================================================================
# Config
# ==============================================================================
## Directory Path
MAKEFILE_DIR := $(dir $(abspath $(lastword $(MAKEFILE_LIST))))
PARENT_DIR   := $(shell dirname $(MAKEFILE_DIR))

# Architecture
## 3bplus/zero2w/4b/5
MODEL          ?= 5
CPU_FOR_3bplus := cortex-a53
CPU_FOR_zero2w := cortex-a53
CPU_FOR_4b     := cortex-a72
CPU_FOR_5      := cortex-a76
TARGET_CPU      = $(CPU_FOR_$(MODEL))

# build
IMAGE_NAME   := picar-builder
TARGET       := aarch64-unknown-linux-gnu
PROJECT_ROOT := $(shell which cygpath > /dev/null 2>&1 && cygpath -w $(PARENT_DIR) || echo "$(PARENT_DIR)")
APP_NAME     := picar-x-rust

# deploy
RPI_HOST   ?= pi@raspberrypi.local
RPI_DIR    ?= /home/pi/

# ==============================================================================
# Commands
# ==============================================================================
.PHONY: all image build fmt lint test deploy clean

all: build

image:
	@echo "Building Docker cross-compile image..."
	docker build -t $(IMAGE_NAME) .

build:
	@echo "Compiling Rust binary for $(TARGET) inside Docker..."
	docker run --rm \
		-v "$(PROJECT_ROOT)":/workspace \
		-v cargo-cache:/usr/local/cargo/registry \
		-v cargo-git:/usr/local/cargo/git \
		-e RUSTFLAGS="-C target-cpu=$(TARGET_CPU)" \
		$(IMAGE_NAME) \
		cargo build --target $(TARGET) --release

fmt:
	@echo "Checking Rust formatting..."
	cargo fmt --all -- --check

lint:
	@echo "Linting Rust code..."
	cargo clippy --target $(TARGET) --all-targets --all-features -- -D warnings

test:
	@echo "Running Rust tests..."
	cargo test --target $(TARGET)

deploy:
	@if [ -z "$(RPI_HOST)" ]; then \
		echo "Error: RPI_HOST is not set. Example: make deploy RPI_HOST=pi@raspberrypi.local"; \
		exit 1; \
	fi
	@echo "Deploying $(APP_NAME) to $(RPI_HOST):$(RPI_DIR)..."
	scp ../target/$(TARGET)/release/$(APP_NAME) $(RPI_HOST):$(RPI_DIR)

clean:
	@echo "Cleaning build artifacts..."
	cargo clean
