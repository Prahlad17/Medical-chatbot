#!/bin/bash

# Define the model path
MODEL_PATH="/app/model/llama-2-7b-chat.ggmlv3.q4_0.bin"

# Check if the model file already exists
if [ ! -f "$MODEL_PATH" ]; then
    echo "Model not found. Downloading..."
    mkdir -p /app/model
    wget --no-cache https://huggingface.co/TheBloke/Llama-2-7B-Chat-GGML/resolve/main/llama-2-7b-chat.ggmlv3.q4_0.bin -O "$MODEL_PATH"
else
    echo "Model already exists. Skipping download."
fi

# Start the application with Gunicorn
gunicorn --bind 0.0.0.0:5000 --workers 1 --timeout 120 app:app
