FROM python:3.11-slim AS builder
WORKDIR /builder

RUN python -m venv /opt/venv

COPY services/api/requirements.txt .
ENV PATH="/opt/venv/bin:$PATH"
RUN pip install --no-cache-dir -r requirements.txt


FROM python:3.11-slim

COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"
ENV PYTHONPATH="/app"

RUN adduser --disabled-password --no-create-home appuser
WORKDIR /app

COPY shared/ shared/
COPY services/api/ services/api/
USER appuser

EXPOSE 8000

CMD ["uvicorn", "services.api.main:app", "--host", "0.0.0.0", "--port", "8000"]