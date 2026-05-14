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

# Run the unified benchmark suite.
benchmark mode='fast' compare='false': build
    #!/usr/bin/env bash
    set -euo pipefail
    raw_mode="{{mode}}"
    raw_compare="{{compare}}"
    mode="${raw_mode#mode=}"
    compare="${raw_compare#compare=}"
    case "${mode}" in
        fast) benchmark_mode="smoke" ;;
        slow) benchmark_mode="standard" ;;
        *) echo "mode must be fast or slow" >&2; exit 2 ;;
    esac
    compare_flag=()
    case "${compare}" in
        true|yes|1) compare_flag=(--benchmark-compare-open3d) ;;
        false|no|0) ;;
        *) echo "compare must be true or false" >&2; exit 2 ;;
    esac
    uv run --group benchmark pytest tests/test_open3d_benchmark.py -v -s --run-slow --benchmark-mode="${benchmark_mode}" "${compare_flag[@]}" --benchmark-json=reports/benchmarks/last-benchmark.json --no-cov

benchmark-visualize:
    uv run --group benchmark python tools/render_open3d_benchmark_charts.py reports/benchmarks/last-benchmark.json --html-output reports/benchmarks/last-benchmark.html --summary-output reports/benchmarks/last-benchmark-summary.csv

benchmark-check baseline='':
    #!/usr/bin/env bash
    set -euo pipefail
    baseline="{{baseline}}"
    if [[ -n "${baseline}" ]]; then
        uv run python tools/check_open3d_speed_budget.py reports/benchmarks/last-benchmark-summary.csv --mode standard --min-1m-speedup 0.80 --min-1m-geomean 1.25 --baseline "${baseline}" --max-regression-ratio 0.10
    else
        uv run python tools/check_open3d_speed_budget.py reports/benchmarks/last-benchmark-summary.csv --mode standard --min-1m-speedup 0.80 --min-1m-geomean 1.25 --no-baseline
    fi

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
