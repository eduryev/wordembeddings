# Copyright 2019 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the \"License\");
# you may not use this file except in compliance with the License.\n",
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an \"AS IS\" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Dockerfile-gpu
# FROM nvidia/cuda:10.2-cudnn7-runtime
FROM tensorflow/tensorflow:nightly-gpu

# Installs necessary dependencies.
RUN apt-get update && apt-get install -y --no-install-recommends \
         wget \
         curl \
         python-dev && \
     rm -rf /var/lib/apt/lists/*

# Installs python3 to run script
RUN apt-get update && apt-get install -y \
	python3 \
	python3-dev \
	python3-pip

# Installs pip.
RUN curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py && \
    python get-pip.py && \
    pip install setuptools && \
    rm get-pip.py
   
WORKDIR /root

# Installs cloudml-hypertune for hyperparameter tuning.
# It’s not needed if you don’t want to do hyperparameter tuning.
# RUN pip install cloudml-hypertune

# Installs google cloud sdk, this is mostly for using gsutil to export model.
RUN wget -nv \
    https://dl.google.com/dl/cloudsdk/release/google-cloud-sdk.tar.gz && \
    mkdir /root/tools && \
    tar xvzf google-cloud-sdk.tar.gz -C /root/tools && \
    rm google-cloud-sdk.tar.gz && \
    /root/tools/google-cloud-sdk/install.sh --usage-reporting=false \
        --path-update=false --bash-completion=false \
        --disable-installation-options && \
    rm -rf /root/.config/* && \
    ln -s /root/.config /config && \
    # Remove the backup directory that gcloud creates
    rm -rf /root/tools/google-cloud-sdk/.install/.backup

# Path configuration
ENV PATH $PATH:/root/tools/google-cloud-sdk/bin
# Make sure gsutil will use the default service account
RUN echo '[GoogleCompute]\nservice_account = default' > /etc/boto.cfg

# update setuptools and pip
RUN pip3 install --upgrade setuptools pip google-api-core

# Install tensorflow 2.2
# RUN pip3 install tensorflow==2.2.0-rc3

# Installs pytorch and torchvision plus other core packages
# RUN pip3 install torch==1.4.0 torchvision==0.5.0 advertorch
# Install smaller packages
RUN pip3 install pyyaml tqdm google-cloud-storage retrying

# Copies the trainer code
#RUN mkdir /root/cifar /root/torchattacks
#    mkdir /root/torchattacks
#    mkdir /root/adv_proj/cifar \
#    mkdir /root/adv_proj/torchattacks
#COPY cifar/ /root/cifar/
#COPY torchattacks/ /root/torchattacks/
COPY . /root/
#RUN echo $(ls cifar/models -LR)

#COPY setup.py /root/
#     torchattacks/* /root/torchattacks/
# adv_proj/cifar/* /root/adv_proj/cifar/ \
# adv_proj/torchattacks/* /root/adv_proj/torchattacks/

# this installs newmodel as a package
RUN pip3 install .

# Sets up the entry point to invoke the trainer.
ENTRYPOINT ["python3", "newmodel/task.py"]
