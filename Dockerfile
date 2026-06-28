FROM python:3.12-slim

WORKDIR /app

# install serving dependencies first so this layer is cached across code changes
COPY requirements-docker.txt .
RUN pip install --no-cache-dir -r requirements-docker.txt

# copy the API code and ONLY the two trained artifacts it loads at runtime
COPY app/ ./app/
COPY artifacts/data_transformation/preprocessor.joblib ./artifacts/data_transformation/preprocessor.joblib
COPY artifacts/model_trainer/model.joblib ./artifacts/model_trainer/model.joblib

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]