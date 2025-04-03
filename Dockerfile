FROM huggingface/transformers-pytorch-gpu:4.41.2

ARG DEBIAN_FRONTEND=noninteractive
WORKDIR /src

COPY requirements.txt $WORKDIR

RUN pip install -U pip setuptools && \
	pip install -r requirements.txt

CMD ["jupyter", "notebook", "--port=8888", "--no-browser", "--ip=0.0.0.0", "--NotebookApp.token=''", "--allow-root"]
