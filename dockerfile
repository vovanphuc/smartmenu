FROM onnx/onnx-ecosystem:latest

WORKDIR /workspace
ENV PYTHONIOENCODING=utf8
COPY ./requirements.txt /workspace
RUN pip install -r /workspace/requirements.txt
COPY . /workspace

CMD ["python3", "api.py"]
