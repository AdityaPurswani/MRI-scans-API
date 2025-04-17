# Use a specific Python version (e.g., 3.9, 3.10, 3.11)
FROM python:3.11

# Create a non-root user 'user'
RUN useradd -m -u 1000 user

# Add the user's local bin directory to the PATH
ENV PATH="/home/user/.local/bin:$PATH"

# Set the working directory
WORKDIR /app

# Copy requirements file
COPY ./requirements.txt requirements.txt

# Install dependencies using --user
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir --user -r requirements.txt

# Switch to the non-root user
USER user

# Re-set WORKDIR context after USER switch
WORKDIR /app

# Copy the application code (contents of local 'app' folder)
COPY ./app/ ./

# Expose the port (Optional - Render handles mapping, but good practice)
# EXPOSE ${PORT:-10000} # You could expose the dynamic port if desired

# ---- MODIFIED CMD for Render ----
# Optional: Update CMD for consistency and local testing
CMD ["gunicorn", "-k", "uvicorn.workers.UvicornWorker", "-b", "0.0.0.0:${PORT:-10000}", "main:app"]
