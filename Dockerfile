# Use an official Python runtime as a parent image
FROM python:3.11-slim

# Environment
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# Copy requirements and install
COPY requirements_dev.txt .
RUN pip install --upgrade pip \
    && pip install --no-cache-dir -r requirements_dev.txt

# Copy application code
COPY src/app /app

# ✅ Copy your trained model into the container
COPY data/models/xgb_pipeline.pkl /data/models/xgb_pipeline.pkl

# ✅ (Optionally) copy other useful folders like templates or static assets
# COPY src/templates /templates
# COPY src/app/static /app/static


# Expose port used by uvicorn
EXPOSE 8000

# Start the app (module path matches how the project is laid out)
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
