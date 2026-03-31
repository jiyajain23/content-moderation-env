FROM python:3.11-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy all files
COPY . .

# Ensure server is a package
RUN touch server/__init__.py

# Set PYTHONPATH so 'server.app' can be found
ENV PYTHONPATH=/app

# Mandatory HF Port
EXPOSE 7860

# Run using the module notation
CMD ["uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "7860"]
