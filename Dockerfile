FROM python:3.9-slim
WORKDIR /app
COPY app/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY app/ .
COPY models/ ./models/
RUN useradd -m -r appuser && chown -R appuser /app
USER appuser
EXPOSE 5000
CMD ["python", "app.py"]
