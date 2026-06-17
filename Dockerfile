# Base image — Python 3.11 slim version
# will use slim verison coz we don't want the os to be packaged and the image size to be small
FROM python:3.11-slim

# To set working directpry in container
WORKDIR /app

# install dependencies  
# build-essential — Some python packages needs to be compiled
# curl — for curl health checks
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*
# rm -rf /var/lib/apt/lists/* — clean cache, decrease the image size 

# To copy requiremenst first 
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
# --no-cache-dir — To not keep pip cache, so that image size will be small 

# Copy the whole project 
COPY . .

# Make docs folder all the pdfs will be stored here 
RUN mkdir -p docs

# Port expose— Streamlit uses default 8501 
EXPOSE 8501

# Health check — To check container working properly or not 
HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health || exit 1

# Run the app 
# --server.address=0.0.0.0 — container should be accesible from outside
# --server.port=8501 — fixed port
# --server.fileWatcherType=none — File watching off in production 
CMD ["streamlit", "run", "app.py", \
     "--server.address=0.0.0.0", \
     "--server.port=8501", \
     "--server.fileWatcherType=none"]