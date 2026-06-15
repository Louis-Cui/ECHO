FROM continuumio/miniconda3:latest

# Create the rgzn conda environment
RUN conda create -n rgzn python=3.10 -y

# Set shell to use the conda environment
SHELL ["conda", "run", "-n", "rgzn", "/bin/bash", "-c"]

WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .
RUN pip install -r requirements.txt -i https://mirrors.aliyun.com/pypi/simple/ \
    && conda clean -afy

# Copy the rest of the application
COPY . .

EXPOSE 8000

# Run with the rgzn environment
CMD ["conda", "run", "-n", "rgzn", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
