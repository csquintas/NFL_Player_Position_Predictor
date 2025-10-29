# ===== Base image =====
FROM python:3.12-slim

# ===== Working directory =====
WORKDIR /app

# ===== Copy requirements first (for caching) =====
COPY requirements_deploy.txt .

# ===== Install dependencies =====
RUN pip install --no-cache-dir -r requirements_deploy.txt

# ===== Copy rest of the code =====
COPY . .

# ===== Expose FastAPI port =====
EXPOSE 8000

# ===== Run the FastAPI app =====
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
