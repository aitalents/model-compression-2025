FROM ultralytics/ultralytics:8.3.89-python
WORKDIR /src

COPY ./ $WORKDIR
RUN pip install jupyter

EXPOSE 8888
ENTRYPOINT ["jupyter", "notebook", "--ip", "0.0.0.0", "--port", "8888", "--allow-root", "--NotebookApp.token=''" ]
