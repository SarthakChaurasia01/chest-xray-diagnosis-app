# Use Python slim image
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Copy all necessary files
COPY requirements.txt .
COPY app.py .
COPY Chest-X-Ray-Classification_Model.h5 .

# Install dependencies
RUN pip install --upgrade pip && \
    pip install -r requirements.txt

# Expose default Gradio port
EXPOSE 7860

# Run the app
CMD ["python", "app.py"]
