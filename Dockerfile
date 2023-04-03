FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . /app

# Install any needed packages specified in requirements.txt
RUN pip install -r requirements.txt

# Make port 80 available to the world outside this container

# Run app.py when the container launches
CMD ["gunicorn", "-w", "2", "--threads", "2", "-b", "0.0.0.0:8080", "main:app"]
