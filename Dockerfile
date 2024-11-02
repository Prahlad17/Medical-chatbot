# Step 1: Use an official Python image as a base
FROM python:3.9-slim

# Step 2: Set the working directory
WORKDIR /app

# Step 3: Install system dependencies, including wget
RUN apt-get update && \
    apt-get install -y wget && \
    rm -rf /var/lib/apt/lists/*

# Step 4: Download the model during build
RUN mkdir -p model && \
    wget --no-cache https://huggingface.co/TheBloke/Llama-2-7B-Chat-GGML/resolve/main/llama-2-7b-chat.ggmlv3.q4_0.bin -O model/llama-2-7b-chat.ggmlv3.q4_0.bin

# Step 5: Copy requirements and install dependencies
COPY requirements.txt /app/
RUN pip install --no-cache-dir -r requirements.txt

# Step 6: Copy all application code
COPY . /app

# Step 7: Expose necessary port
EXPOSE 5000

# Step 8: Run the application
CMD ["python", "app.py"]
