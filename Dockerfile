# Step 1: Use an official Python image as a base
FROM python:3.9-slim

# Step 2: Set the working directory
WORKDIR /app

# Step 3: Install system dependencies, including wget
RUN apt-get update && \
    apt-get install -y wget && \
    pip install --upgrade pip setuptools && \
    rm -rf /var/lib/apt/lists/*

# Step 4: Copy requirements and install dependencies
COPY requirements.txt /app/
RUN pip install --no-cache-dir -r requirements.txt && \
    pip install gunicorn  

# Step 5: Copy all application code
COPY . /app

# Step 6: Expose necessary port
EXPOSE 5000

# Step 7: Set up the entrypoint script to download the model and start the app
COPY entrypoint.sh /app/entrypoint.sh
RUN chmod +x /app/entrypoint.sh

# Step 8: Run the entrypoint script
CMD ["/app/entrypoint.sh"]
