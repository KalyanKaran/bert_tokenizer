# Use the official lightweight Python image.
FROM python:3.9-slim

# Set the working directory inside the container.
WORKDIR /app

# Copy the requirements file and install dependencies
COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

# Install Accelerate explicitly
RUN pip install "transformers[torch]" "accelerate>=0.26.0"

# Copy the rest of the application code
COPY . .

# Verify installation (debugging step)
RUN python -c "import accelerate; print('Accelerate Installed Successfully')"


# Expose a port (useful if running Jupyter Notebook or similar).
EXPOSE 8888

# Set the default command to run your training script from the correct directory.
CMD ["python", "/app/train.py"]