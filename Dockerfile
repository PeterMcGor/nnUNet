# Contains pytorch, torchvision, cuda, cudnn
FROM nvcr.io/nvidia/pytorch:22.04-py3
MAINTAINER Pedro M. Gordaliza @petermcgor

#nnUnet is set up by default to look at the following dir. I mantain them for reproducibility
ARG resources="/opt/nnunet_resources"
ENV nnUNet_raw_data_base=$resources nnUNet_preprocessed=$resources"/nnUNet_preprocessed" RESULTS_FOLDER=$resources"/nnUNet_trained_models"

#Copy the files. TODO change to git clone https://github.com/PeterMcGor/nnUNet.git  once the repo is tested the repo is tested
RUN mkdir /nnunet
WORKDIR /nnunet
COPY . .

## Install nnunet 
RUN pip install -e .
## hiddenlayers for topologies 
RUN pip install --upgrade git+https://github.com/FabianIsensee/hiddenlayer.git@more_plotted_details#egg=hiddenlayer
