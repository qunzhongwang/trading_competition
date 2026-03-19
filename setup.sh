if ! command -v uv &> /dev/null; then
    pip install uv
fi
uv sync
pre-commit install
