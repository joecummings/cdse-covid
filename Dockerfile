# Dockerfile, Image, Container
# Before building, please run setup.sh to download necessary files (?)
FROM ubuntu:latest

LABEL name="ISI Claim Detection and Semantic Extraction COVID-19"
LABEL version=0
LABEL maintainer="cummings@isi.edu"

ENV PATH /opt/conda/bin:$PATH

# Ubuntu timezone config
ENV TZ=America/New_York
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

ARG isi_username=""

# Install base packages
RUN apt-get update --fix-missing && apt-get install -y \
    cmake \
    git \
    software-properties-common \
    unzip \
    wget

# Install Java 8
RUN add-apt-repository -y \
    ppa:webupd8team/java

RUN apt-get update && \
    apt-get install -y openjdk-8-jdk && \
    apt-get install -y ant && \
    apt-get clean;

# Fix certificate issues
RUN apt-get update && \
    apt-get install ca-certificates-java && \
    apt-get clean && \
    update-ca-certificates -f && \
    rm -rf /var/lib/apt/lists/* && \
    rm -rf /var/cache/oracle-jdk8-installer

ENV JAVA_HOME /usr/lib/jvm/java-8-openjdk-amd64/
RUN export JAVA_HOME

# Install miniconda
RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh && \
    /bin/bash ~/miniconda.sh -b -p /opt/conda && \
    rm ~/miniconda.sh && \
    ln -s /opt/conda/etc/profile.d/conda.sh /etc/profile.d/conda.sh && \
    echo ". /opt/conda/etc/profile.d/conda.sh" >> ~/.bashrc

# Main conda env
ADD ./cdse_covid /cdse-covid/cdse_covid
ADD ./wikidata_linker /cdse-covid/wikidata_linker
ADD ./requirements-lock.txt /cdse-covid/requirements-lock.txt
RUN /opt/conda/bin/conda create -n cdse-covid python=3.7 && \
    /opt/conda/envs/cdse-covid/bin/pip install -r /cdse-covid/requirements-lock.txt && \
    /opt/conda/envs/cdse-covid/bin/python -m spacy download en_core_web_md && \
    /opt/conda/envs/cdse-covid/bin/python -m nltk.downloader -d /opt/conda/envs/cdse-covid/nltk_data wordnet && \
    /opt/conda/envs/cdse-covid/bin/python -m nltk.downloader -d /opt/conda/envs/cdse-covid/nltk_data framenet_v17 && \
    /opt/conda/envs/cdse-covid/bin/python -m nltk.downloader -d /opt/conda/envs/cdse-covid/nltk_data stopwords

# Transition AMR Parser conda env
ADD ./amr-requirements-lock.txt /cdse-covid/amr-requirements-lock.txt
RUN /opt/conda/bin/conda create -n transition-amr-parser python=3.7 && \
    /opt/conda/envs/transition-amr-parser/bin/pip install -r /cdse-covid/amr-requirements-lock.txt && \
    /opt/conda/envs/transition-amr-parser/bin/python -m spacy download en_core_web_md && \
    /opt/conda/envs/transition-amr-parser/bin/python -m nltk.downloader -d /opt/conda/envs/cdse-covid/nltk_data wordnet && \
    /opt/conda/envs/transition-amr-parser/bin/python -m nltk.downloader -d /opt/conda/envs/cdse-covid/nltk_data framenet_v17 && \
    /opt/conda/envs/transition-amr-parser/bin/python -m nltk.downloader -d /opt/conda/envs/cdse-covid/nltk_data stopwords

# Download sentence model weights
RUN wget https://public.ukp.informatik.tu-darmstadt.de/reimers/sentence-transformers/v0.2/stsb-roberta-base.zip && \
    unzip stsb-roberta-base.zip -d stsb-roberta-base && \
    mv stsb-roberta-base /cdse-covid/wikidata_linker/sent_model/ && \
    rm stsb-roberta-base.zip

# Create KGTK cache
RUN mkdir /cdse-covid/wikidata_linker/kgtk_event_cache
RUN mkdir /cdse-covid/wikidata_linker/kgtk_refvar_cache

# Download wikidata classifier
# TBA

# Install the transition AMR parser
RUN git clone https://github.com/IBM/transition-amr-parser.git && \
    cd /transition-amr-parser && \
    git checkout action-pointer && \
    touch set_environment.sh && \
    /opt/conda/envs/transition-amr-parser/bin/python -m pip install -e . && \
    sed -i.bak "s/pytorch\/fairseq'/\pytorch\/fairseq\:main'/" transition_amr_parser/parse.py

# Install the JAMR aligner
RUN cd /transition-amr-parser/preprocess && \
    git clone https://github.com/jflanigan/jamr.git && \
    mkdir /.sbt && \
    printf "[repositories]\n\tmaven-central: https://repo1.maven.org/maven2/" > /.sbt/repositories && \
    cd jamr && \
    git checkout Semeval-2016 && \
    grep -v "import AssemblyKeys._" "build.sbt" > tmpfile && mv tmpfile "build.sbt" && \
    grep -v "assemblySettings" "build.sbt" > tmpfile && mv tmpfile "build.sbt" && \
    grep -v "sbt-idea" "project/plugins.sbt" > tmpfile && mv tmpfile "project/plugins.sbt" && \
    sed -i.bak "s/\"scala-arm\" % \"[0-9]*\.[0-9]*\"/\"scala-arm\" % \"2\.0\"/" "build.sbt" && \
    sed -i.bak "s/\"sbt-assembly\" % \"[0-9]*\.[0-9]*\.[0-9]*\"/\"sbt-assembly\" % \"0\.14\.6\"/" "project/plugins.sbt" && \
    sed -i.bak "s/\"sbteclipse-plugin\" % \"[0-9]*\.[0-9]*\.[0-9]*\"/\"sbteclipse-plugin\" % \"5\.2\.4\"/" "project/plugins.sbt" && \
    /transition-amr-parser/preprocess/jamr/setup && \
    /transition-amr-parser/preprocess/jamr scripts/config.sh

# Install the Kevin aligner
RUN cd /transition-amr-parser/preprocess && \
    git clone https://github.com/damghani/AMR_Aligner && \
    mv AMR_Aligner kevin && \
    cd kevin && \
    git clone https://github.com/moses-smt/mgiza.git && \
    cd mgiza && \
    /opt/conda/envs/transition-amr-parser/bin/cmake . && \
    /opt/conda/envs/transition-amr-parser/bin/make && \
    make install \
    cd /

# Download AMR parser model
# TBA

CMD ["/bin/bash"]
