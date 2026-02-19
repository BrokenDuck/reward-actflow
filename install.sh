#!/usr/bin/env bash
if command -v module >/dev/null 2>&1
then
    module load stack/2024-04 gcc/8.5.0 cuda/11.3.1
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

if command -v uv >/dev/null 2>&1
then
    uv pip install 'openfold @ git+https://github.com/aqlaboratory/openfold.git@4b41059694619831a7db195b7e0988fc4ff3a307' --no-build-isolation
else
    pip install 'openfold @ git+https://github.com/aqlaboratory/openfold.git@4b41059694619831a7db195b7e0988fc4ff3a307' --no-build-isolation
fi


cd $MAIN_DIR
if  [ ! -d "sgpo" ]
then
    git clone git@github.com:Komod0D/SGPO.git sgpo
    cd sgpo/sgpo
    mkdir -p checkpoints/continuous_ESM/CreiLOV
    cd checkpoints/continuous_ESM/CreiLOV
    wget https://huggingface.co/jsunn-y/SGPO/resolve/main/checkpoints/continuous_ESM/CreiLOV/best_model.ckpt
    cd $MAIN_DIR/sgpo
    if command -v uv >/dev/null 2>&1
        then
            uv pip install -e . --config-settings editable_mode=strict
        else
            pip install -e . --config-settings editable_mode=strict
    fi
    cd sgpo
    python fix_checkpoints.py
fi