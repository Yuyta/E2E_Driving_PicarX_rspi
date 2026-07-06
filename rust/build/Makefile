# ==============================================================================
# Config
# ==============================================================================
## Directory Path
MAKEFILE_DIR :=$(dir $(abspath $(lastword $(MAKEFILE_LIST))))
PARENT_DIR   := $(shell dirname ${MAKEFILE_DIR})

# Architecture
## 3bplus/zero2w/4b/5
MODEL          := 5
CPU_FOR_3bplus := cortex-a53
CPU_FOR_zero2w := cortex-a53
CPU_FOR_4b     := cortex-a72
CPU_FOR_5      := cortex-a76
TARGET_CPU      = $(CPU_FOR_$(MODEL))

# build
IMAGE_NAME   := picar-builder
TARGET       := aarch64-unknown-linux-gnu
# PROJECT_ROOT := $(PARENT_DIR)
PROJECT_ROOT := C:\Users\tacta\Dev\PiCarX\rust
APP_NAME     := picar-x-rust

# deploy
RPI_IP     := 192.168.40.119
RPI_USER   := y_murazato
RPI_DIR    := /home/y_murazato

# ==============================================================================
# Commands
# ==============================================================================
.PHONY: all image build deploy clean

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

deploy:
	@echo "Sending $(APP_NAME) to Raspberry Pi ($(RPI_IP))..."
	scp $(PROJECT_ROOT)/target/$(TARGET)/release/$(APP_NAME) $(RPI_USER)@$(RPI_IP):$(RPI_DIR)/
	@echo "Deployment complete! Run './$(APP_NAME)' on your Raspberry Pi."

clean:
	@echo "Cleaning up target directory..."
	cargo clean

