# Default recipe - show available commands
default:
    @just --list

# Install development dependencies
install:
    uv sync --all-groups
    pre-commit install

# Build Rust extension in development mode
dev:
    uv run maturin develop

# Build Rust extension in release mode
build:
    uv run maturin develop --release

test:
    just dev
    uv run pytest tests/ -v

test-slow:
    uv run pytest tests/ -v --run-slow

# Run benchmark tests
benchmark: benchmark-smoke

benchmark-smoke: build
    uv run pytest tests/test_benchmark.py -v -s --run-slow --benchmark-mode=smoke --no-cov

benchmark-standard: build
    uv run pytest tests/test_benchmark.py -v -s --run-slow --benchmark-mode=standard --no-cov

benchmark-full: build
    uv run pytest tests/test_benchmark.py -v -s --run-slow --benchmark-mode=full --no-cov

benchmark-docs:
    uv run python tools/render_benchmark_docs.py

benchmark-compare-smoke: build
    uv run --group benchmark pytest tests/test_open3d_benchmark.py -v -s --run-slow --benchmark-mode=smoke --benchmark-json=reports/benchmarks/open3d-comparison-smoke.json --no-cov

benchmark-compare-standard: build
    uv run --group benchmark pytest tests/test_open3d_benchmark.py -v -s --run-slow --benchmark-mode=standard --benchmark-json=reports/benchmarks/open3d-comparison-standard.json --no-cov

benchmark-compare-full: build
    uv run --group benchmark pytest tests/test_open3d_benchmark.py -v -s --run-slow --benchmark-mode=full --benchmark-json=reports/benchmarks/open3d-comparison-full.json --no-cov

benchmark-compare-charts:
    uv run --group benchmark python tools/render_open3d_benchmark_charts.py

# Run Rust tests
test-rust:
    cargo test --release

# Format code (Rust and Python)
fmt:
    cargo fmt
    uv run ruff format

# Lint code (Rust and Python)
lint:
    cargo clippy -- -D warnings
    uv run ruff check

# Run pre-commit on all files
pre-commit:
    pre-commit run --all-files

# Clean build artifacts
clean:
    cargo clean
    rm -rf target/
    rm -rf dist/
    rm -rf *.egg-info
    find . -type d -name __pycache__ -exec rm -rf {} +
    find . -type f -name "*.pyc" -delete

# Build wheel packages
wheel:
    uv build --wheel

# Build source and wheel distributions
dist:
    uv build

# Serve documentation locally
docs-serve:
    uv run mkdocs serve

# Build documentation
docs-build:
    uv run mkdocs build

# Deploy documentation to GitHub Pages
docs-deploy:
    uv run mkdocs gh-deploy --force

# Release workflow: format, lint, test, build
release: fmt lint test build wheel
    @echo "✅ Release checks passed!"

# CI workflow: all checks
ci: test-rust test
    @echo "✅ CI checks passed!"
