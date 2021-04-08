# pull official python alpine image
FROM ubuntu:latest
# Define global args
ARG FUNCTION_DIR="/home/app/"

# Making source and static directory
RUN mkdir /src
RUN mkdir /static

# Include global args in this stage of the build
ARG FUNCTION_DIR
ARG RUNTIME_VERSION
# Create function directory
RUN mkdir -p /home/app/
# Copy handler function
COPY app.py /home/app/

# Creating Work Directory
WORKDIR /src

RUN apt-get update \
  && apt-get install -y python3-pip python3-dev \
  && cd /usr/local/bin \
  && ln -s /usr/bin/python3 python \
  && pip3 install --upgrade pip

# Update pip
RUN pip install --upgrade pip

# Installing requirements.pip from project
COPY ./requirements.txt /scripts/
RUN pip install --no-cache-dir -r /scripts/requirements.txt

RUN pip install awslambdaric --target /home/app/

COPY glove.6B.100d-new.sav /home/app/glove.6B.100d_model.sav

WORKDIR /home/app/
# (Optional) Add Lambda Runtime Interface Emulator and use a script in the ENTRYPOINT for simpler local runs
ADD https://github.com/aws/aws-lambda-runtime-interface-emulator/releases/latest/download/aws-lambda-rie /usr/bin/aws-lambda-rie
COPY entry.sh /
RUN chmod 755 /usr/bin/aws-lambda-rie /entry.sh
ENTRYPOINT [ "/entry.sh" ]
CMD [ "app.handler" ]