#!/usr/bin/env bash
if command -v module >/dev/null 2>&1
then
    module load stack/2024-06 cuda/12.1.1
fi

ENV_DIR=$(pwd)
INSTALL_DIR=$1
cd $INSTALL_DIR
mkdir -p .local
cd .local
MAIN_DIR=$(pwd)
mkdir -p bin

if ! command -v mafft >/dev/null 2>&1
then 
    wget https://mafft.cbrc.jp/alignment/software/mafft-7.525-with-extensions-src.tgz
    tar xfvz mafft-7.525-with-extensions-src.tgz
    rm mafft-7.525-with-extensions-src.tgz
    mv mafft* mafft
    cd mafft
    echo "PREFIX=$MAIN_DIR" > temp
    cat core/Makefile | tail +2 >> temp
    mv temp core/Makefile
    cd core
    make clean
    make
    make install
    cd $MAIN_DIR
fi

if ! command -v hhalign >/dev/null 2>&1
then
    cd $MAIN_DIR
    mkdir -p hhsuite 
    cd hhsuite
    wget https://github.com/soedinglab/hh-suite/releases/download/v3.3.0/hhsuite-3.3.0-AVX2-Linux.tar.gz
    tar xvfz hhsuite-3.3.0-AVX2-Linux.tar.gz
    echo "export PATH=$(pwd)/bin:$(pwd)/scripts:\$PATH" >> ~/.bashrc
    cd $MAIN_DIR
fi

if ! command -v mmseqs >/dev/null 2>&1
then
    cd $MAIN_DIR
    wget https://mmseqs.com/latest/mmseqs-linux-gpu.tar.gz
    tar xvfz mmseqs-linux-gpu.tar.gz
    rm mmseqs-linux-gpu.tar.gz
    echo "export PATH=$(pwd)/mmseqs/bin/:\$PATH" >> ~/.bashrc
    cd $MAIN_DIR
fi


cd $MAIN_DIR
if [ ! -d "openfold" ]
then
    git clone https://github.com/aqlaboratory/openfold.git
    cd openfold
    cat scripts/install_third_party_dependencies.sh | head -n 17 > scripts/our_install.sh
    chmod +x scripts/our_install.sh
    ./scripts/our_install.sh
    echo "export CUTLASS_PATH=$(pwd)/cutlass" >> ~/.bashrc
    echo "export KMP_AFFINITY=none" >> ~/.bashrc
    echo "export LIBRARY_PATH=$ENV_DIR/.venv/lib:\$LIBRARY_PATH" >> ~/.bashrc
    echo "export LD_LIBRARY_PATH=$ENV_DIR/.venv/lib:\$LD_LIBRARY_PATH" >> ~/.bashrc
fi
