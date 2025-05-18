FROM huggingface/transformers-pytorch-gpu:4.35.0
ARG DEBIAN_FRONTEND=noninteractive

WORKDIR /src
ENV PYTHONPATH="${PYTHONPATH}:${WORKDIR}"
ENV TRANSFORMERS_CACHE="/src/.cache/"

COPY requirements.txt $WORKDIR
COPY quant_models.sh $WORKDIR

RUN apt-get update && apt upgrade -y && \
		apt-get install -y libsm6 libxrender1 libfontconfig1 libxext6 libgl1-mesa-glx ffmpeg && \
		pip install -U pip setuptools && \
		pip install -U --no-cache-dir -r requirements.txt

# try this if the above fails

# RUN apt-get update && \
# 		apt-get install -y --no-install-recommends libsm6 libxrender1 libfontconfig1 libxext6 libgl1-mesa-glx ffmpeg && \
# 		pip install -U pip setuptools && \
# 		pip install -U --no-cache-dir -r requirements.txt

# RUN chmod +x quant_models.sh && \
# 		./quant_models.sh

COPY . $WORKDIR

ENTRYPOINT [ "uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8080", "--workers", "1" ]
