FROM python:3.7

# Create directories
RUN mkdir -p /app/src/
RUN mkdir -p /app/data/
RUN mkdir -p /app/model/

# Copy training code
COPY main.py /app/src/
COPY requirements.txt /app/src/

# Copy dataset (from datasets folder)
COPY ../datasets/train.csv /app/data/train.csv

# Install dependencies
RUN pip3 install --no-cache-dir -r /app/src/requirements.txt

# Set working directory
WORKDIR /app/src/

# Permissions (important for SAP AI Core)
RUN chgrp -R 65534 /app && chmod -R 777 /app

# Run training
CMD ["python", "main.py"]