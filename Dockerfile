FROM python:3.10-slim
WORKDIR /app

COPY app/requirements.txt ./
COPY app/demo.py ./

RUN pip install --upgrade pip
RUN pip install -r requirements.txt 
RUN curl -fsSL --no-progress-meter https://ollama.com/install.sh | sh
RUN ollama serve

EXPOSE 8501
ENTRYPOINT ["streamlit", "run", "demo.py"]
