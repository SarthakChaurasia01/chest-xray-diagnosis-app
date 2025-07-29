FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .
COPY app.py .
COPY Chest-X-Ray-Classification_Model.h5 .

RUN pip install --upgrade pip && \
    pip install -r requirements.txt

CMD ["python", "app.py"]
