FROM python

WORKDIR /app

# dependencies
COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt

# model
COPY model.py model.py
# startup.py loads and downloads the model files
COPY startup.py startup.py
# run it once so that all the files are available in the image
RUN python startup.py

# files only for serving
COPY server.py server.py

CMD [ "python", "server.py" ]
