FROM python:3.11-slim AS builder
WORKDIR /builder

RUN python -m venv /opt/venv

COPY services/worker/requirements.txt .
ENV PATH="/opt/venv/bin:$PATH"
RUN pip install --no-cache-dir -r requirements.txt

FROM python:3.11-slim

COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"
ENV PYTHONPATH="/app"

RUN adduser --disabled-password --no-create-home workeruser
WORKDIR /app

COPY shared/ shared/
COPY services/worker/ services/worker/
USER workeruser

CMD ["python", "-m", "services.worker.worker"]