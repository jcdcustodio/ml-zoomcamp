# Base Docker image
FROM python:3.14.8-slim

# Copy the 'uv' and 'uvx' executables from the latest uv image into /bin/ in this image
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# Set the working directory inside the container to /src
# All subsequent commands will be run from here
WORKDIR /src

# Add the virtual environment's bin directory to the PATH so Python tools work globally
ENV PATH="/src/.venv/bin:$PATH"

# Copy the project configuration files into the container
# Done first for better layer caching
COPY "pyproject.toml" "uv.lock" ".python-version" ./

# Install dependencies exactly as locked in uv.lock, without updating them
RUN uv sync --locked --no-install-project

# Copy application code and model data into the container
COPY "predict.py" "model_final.bin" ./

# Expose TCP port so it can be accessed from outside the container
EXPOSE 7676

# Run the application using uvicorn (ASGI server)
# predict:app → refers to 'app' object inside predict.py
# --host 0.0.0.0 → listen on all interfaces
# --port 7676    → listen on port 7676
ENTRYPOINT ["uvicorn", "predict:app", "--host", "0.0.0.0", "--port", "7676"]