FROM python:3.10-slim
WORKDIR /app
COPY . .
RUN pip install flask scikit-learn joblib
EXPOSE 5000
CMD ["python", "app.py"]
