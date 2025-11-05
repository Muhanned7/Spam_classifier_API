FROM python:3.10.11


WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY ml_api.py .
COPY logistic_regression_model.joblib .
COPY tfidf_vectorizer.pkl .

EXPOSE 8000

CMD ["gunicorn", "--bind", "0.0.0.0:${PORT}", "ml_api:app"]