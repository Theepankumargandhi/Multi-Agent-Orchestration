FROM python:3.12.11-slim AS runtime

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

COPY requirements-app.txt .
RUN python -m pip install --no-cache-dir -r requirements-app.txt

COPY client/ ./client/
COPY schema/ ./schema/
COPY streamlit_app.py .

RUN addgroup --system app && adduser --system --ingroup app --uid 10001 app \
    && chown -R app:app /app
USER app

EXPOSE 8501
HEALTHCHECK --interval=30s --timeout=3s --start-period=15s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8501/_stcore/health', timeout=2)"

CMD ["streamlit", "run", "streamlit_app.py"]
