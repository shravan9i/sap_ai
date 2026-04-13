FROM python:3.7

# Create directories
RUN mkdir -p /app/src/
RUN mkdir -p /app/model/

# Copy serving code
COPY app.py /app/src/
COPY requirements.txt /app/src/

# ⚠️ Copy trained model (IMPORTANT)
# This assumes model.pkl exists in training folder after training
COPY ../training/model.pkl /app/model/model.pkl

# Install dependencies
RUN pip3 install --no-cache-dir -r /app/src/requirements.txt

# Set working directory
WORKDIR /app/src/

# Permissions (SAP AI Core friendly)
RUN chgrp -R 65534 /app && chmod -R 777 /app

# Start API
CMD ["python", "app.py"]