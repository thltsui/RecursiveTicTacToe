FROM python:3.11-slim

WORKDIR /app

# Install uv for fast dependency management
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# Copy config and lockfiles
COPY pyproject.toml uv.lock ./

# Install project dependencies including web extras
RUN uv sync --frozen --extra web

# Copy source files
COPY 01_game/ ./01_game/
COPY 02_network/ ./02_network/
COPY 03_mcts/ ./03_mcts/
COPY transformer/ ./transformer/
COPY web_app/ ./web_app/

# .dockerignore admits only promoted deployment checkpoints. The web loader
# prefers transformer_best.pt and retains best_ever_model.pt as rollback.
COPY checkpoints/ ./checkpoints/

EXPOSE 5001

# Run the Flask-SocketIO app using its built-in eventlet WSGI server
CMD ["uv", "run", "python", "web_app/app.py"]
