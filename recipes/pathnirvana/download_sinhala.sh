#!/bin/bash
# take the scripts's parent's directory to prefix all the output paths.
RUN_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
echo $RUN_DIR
# download Path Nirvana Sinhala dataset
wget https://github.com/pnfo/sinhala-tts-dataset/releases/download/v2.1/sinhala_dataset.tar.bz2
# extract
mkdir sinhala_dataset
tar -xjf sinhala_dataset.tar.bz2 --directory sinhala_dataset
# create train-val splits
shuf sinhala_dataset/metadata.csv > sinhala_dataset/metadata_shuf.csv

mv sinhala_dataset $RUN_DIR/recipes/pathnirvana/
rm sinhala_dataset.tar.bz2