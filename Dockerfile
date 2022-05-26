# Please see the README about instructions before building
FROM nvidia/cuda:10.0-cudnn7-runtime-ubuntu18.04

LABEL name="ISI Claim Detection and Semantic Extraction COVID-19"
LABEL version=0
LABEL maintainer="elee@isi.edu"

ENV PATH /opt/conda/bin:$PATH

# Ubuntu timezone config
ENV TZ=America/New_York
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

# Install base packages
RUN rm /etc/apt/sources.list.d/cuda.list && apt-get update --fix-missing && apt-get install -y \
    cmake \
    g++ \
    git \
    software-properties-common \
    unzip \
    vim \
    wget

# Install miniconda
RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh && \
    /bin/bash ~/miniconda.sh -b -p /opt/conda && \
    rm ~/miniconda.sh && \
    ln -s /opt/conda/etc/profile.d/conda.sh /etc/profile.d/conda.sh && \
    echo ". /opt/conda/etc/profile.d/conda.sh" >> ~/.bashrc

# Main conda env
ENV HOME=/home
COPY ./cdse_covid /cdse-covid/cdse_covid
COPY ./setup.py /cdse-covid/setup.py
COPY ./wikidata_linker /cdse-covid/wikidata_linker
COPY ./requirements-docker-lock.txt /cdse-covid/requirements-docker-lock.txt
COPY ./aida-tools /aida-tools
COPY ./amr-utils /amr-utils
COPY ./saga-tools /saga-tools
RUN /opt/conda/bin/conda create -n cdse-covid python=3.7 && \
    /opt/conda/bin/conda install -c conda-forge jsonnet==0.17.0 && \
    /opt/conda/envs/cdse-covid/bin/pip install -r /cdse-covid/requirements-docker-lock.txt && \
    /opt/conda/envs/cdse-covid/bin/python -m spacy download en_core_web_sm && \
    /opt/conda/envs/cdse-covid/bin/python -m spacy download en_core_web_md && \
    /opt/conda/envs/cdse-covid/bin/python -m nltk.downloader -d /opt/conda/envs/cdse-covid/nltk_data wordnet && \
    /opt/conda/envs/cdse-covid/bin/python -m nltk.downloader -d /opt/conda/envs/cdse-covid/nltk_data wordnet_ic && \
    /opt/conda/envs/cdse-covid/bin/python -m nltk.downloader -d /opt/conda/envs/cdse-covid/nltk_data sentiwordnet && \
    /opt/conda/envs/cdse-covid/bin/python -m nltk.downloader -d /opt/conda/envs/cdse-covid/nltk_data framenet_v17 && \
    /opt/conda/envs/cdse-covid/bin/python -m nltk.downloader -d /opt/conda/envs/cdse-covid/nltk_data stopwords && \
    chmod -R a+rw /opt/conda/envs/cdse-covid && \
    /opt/conda/envs/cdse-covid/bin/pip install /cdse-covid && \
    /opt/conda/envs/cdse-covid/bin/pip install /aida-tools && \
    /opt/conda/envs/cdse-covid/bin/pip install /amr-utils && \
    /opt/conda/envs/cdse-covid/bin/pip install /saga-tools

ENV PYTHONPATH /cdse-covid

# Transition AMR Parser conda env
COPY ./amr-requirements-docker-lock.txt /cdse-covid/amr-requirements-docker-lock.txt
COPY ./transition-amr-parser /transition-amr-parser
COPY ./tap_environment_for_docker.yml ./cdse-covid/tap_environment_for_docker.yml
RUN /opt/conda/bin/conda env create -n transition-amr-parser -f /cdse-covid/tap_environment_for_docker.yml && \
    /opt/conda/envs/transition-amr-parser/bin/pip install -r /cdse-covid/amr-requirements-docker-lock.txt && \
    /opt/conda/envs/transition-amr-parser/bin/python -m spacy download en_core_web_sm && \
    /opt/conda/envs/transition-amr-parser/bin/python -m spacy download en_core_web_md && \
    /opt/conda/envs/transition-amr-parser/bin/python -m nltk.downloader -d /opt/conda/envs/transition-amr-parser/nltk_data wordnet && \
    /opt/conda/envs/transition-amr-parser/bin/python -m nltk.downloader -d /opt/conda/envs/transition-amr-parser/nltk_data framenet_v17 && \
    /opt/conda/envs/transition-amr-parser/bin/python -m nltk.downloader -d /opt/conda/envs/transition-amr-parser/nltk_data stopwords && \
    chmod -R a+rw /opt/conda/envs/transition-amr-parser && \
    /opt/conda/envs/transition-amr-parser/bin/pip install /amr-utils && \
    /opt/conda/envs/transition-amr-parser/bin/pip install /cdse-covid && \
    /opt/conda/envs/transition-amr-parser/bin/pip install /saga-tools

# Create torch env
ENV TORCH_HOME=/cdse-covid/.cache/torch
WORKDIR $TORCH_HOME/hub
WORKDIR /

# Download sentence model weights and fairseq model
RUN wget https://public.ukp.informatik.tu-darmstadt.de/reimers/sentence-transformers/v0.2/stsb-roberta-base.zip && \
    unzip stsb-roberta-base.zip -d stsb-roberta-base && \
    mv stsb-roberta-base /cdse-covid/wikidata_linker/sent_model/ && \
    rm stsb-roberta-base.zip && \
    wget https://github.com/pytorch/fairseq/archive/main.zip && \
    unzip main.zip -d $TORCH_HOME/hub && \
    mv $TORCH_HOME/hub/fairseq-main $TORCH_HOME/hub/pytorch_fairseq_main && \
    rm main.zip

# Create KGTK cache
WORKDIR /cdse-covid/wikidata_linker/kgtk_event_cache
WORKDIR /cdse-covid/wikidata_linker/kgtk_refvar_cache

# Install the transition AMR parser
WORKDIR /transition-amr-parser
RUN git checkout action-pointer && \
    touch set_environment.sh && \
    /opt/conda/envs/transition-amr-parser/bin/python -m pip install -e . && \
    sed -i.bak "s/pytorch\/fairseq'/\pytorch\/fairseq\:main'/" transition_amr_parser/parse.py

WORKDIR /

CMD ["/bin/bash"]
