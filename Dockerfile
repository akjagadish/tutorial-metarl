From sinzlab/pytorch:v3.8-torch1.7.0-cuda11.0-dj0.12.7 

# Install essential Ubuntu packages
# and upgrade pip
RUN apt update &&\
    apt install -y build-essential \
                   apt-transport-https \
                   lsb-release  \
                   ca-certificates \
                   curl

# Update nodejs
RUN curl -sL https://deb.nodesource.com/setup_12.x | bash -
RUN apt install -y nodejs

# install third-party libraries
RUN pip3 install --upgrade pip
RUN pip3 --no-cache-dir install \
         hiplot \
         slacker\
         ax-platform \ 
         tensorboard \
         ptvsd \
         jupyterlab>=2 \
         xeus-python


RUN jupyter labextension install @jupyterlab/debugger

# install the current project as a library
RUN mkdir /notebooks/tutorial_metarl

COPY tutorial_metarl /notebooks/tutorial_metarl
COPY setup.py /notebooks

RUN python3 setup.py develop