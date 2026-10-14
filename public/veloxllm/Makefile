# VeloxLLM Makefile
# Simple build commands like llama.cpp

.PHONY: all release debug clean install bench

# Default target
all: release

# Release build (optimized)
release:
	@echo "⚡ Building VeloxLLM (release mode)..."
	cargo build --release
	@echo "✅ Binary: target/release/veloxllm"

# Debug build
debug:
	@echo "🔧 Building VeloxLLM (debug mode)..."
	cargo build
	@echo "✅ Binary: target/debug/veloxllm"

# Install to ~/.local/bin
install: release
	@echo "📦 Installing VeloxLLM..."
	mkdir -p ~/.local/bin
	cp target/release/veloxllm ~/.local/bin/
	@echo "✅ Installed to ~/.local/bin/veloxllm"
	@echo "   Make sure ~/.local/bin is in your PATH"

# Run benchmarks
bench: release
	./target/release/veloxllm bench $(MODEL)

# Clean build artifacts
clean:
	cargo clean
	@echo "🧹 Cleaned"

# Run with a model
run: release
	./target/release/veloxllm run $(MODEL) --prompt "$(PROMPT)"

# Start server
server: release
	./target/release/veloxllm server $(MODEL) --port $(PORT)

# Format code
fmt:
	cargo fmt

# Run linter
lint:
	cargo clippy -- -D warnings

# Run tests
test:
	cargo test

# Show help
help:
	@echo "VeloxLLM Build Commands:"
	@echo "  make              - Build release binary"
	@echo "  make release      - Build optimized binary"
	@echo "  make debug        - Build debug binary"
	@echo "  make install      - Install to ~/.local/bin"
	@echo "  make run MODEL=x  - Run with model (PROMPT='Hello')"
	@echo "  make server MODEL=x PORT=8080 - Start API server"
	@echo "  make bench MODEL=x - Run benchmarks"
	@echo "  make clean        - Clean build artifacts"
	@echo "  make test         - Run tests"
	@echo "  make fmt          - Format code"
	@echo "  make lint         - Run linter"
