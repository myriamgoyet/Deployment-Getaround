# Use an official lightweight image as the base
FROM python:3.9.18-slim

# Set the working directory in the container
WORKDIR /app

# Copy the necessary files into the container
COPY requirements.txt .

# Upgrade pip
RUN pip install --upgrade pip

# Install the dependencies listed in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt
RUN streamlit --version   

ENV HOME=/home/user

# Copy the application code and config
RUN mkdir -p $HOME/.streamlit
COPY .streamlit/config.toml $HOME/.streamlit/config.toml
COPY app.py .

# Exposeport used by streamlit
EXPOSE 7860

# Define the command to run when the container starts
CMD streamlit run app.py --server.address=0.0.0.0 --server.port=7860