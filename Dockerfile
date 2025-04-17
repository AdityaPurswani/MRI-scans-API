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
# Use the $PORT variable provided by Render, defaulting to 10000 otherwise.
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "${PORT:-10000}"]