#!/bin/bash

# Load fine-tuned model
# Models for two or four languages. Here se stands for Swedish. ;)

mkdir -p models
#model=xlmr-base-fi-se.pt
model=xlmr-base-en-fi-fr-se.pt
wget http://dl.turkunlp.org/register-labeling-model/$model.gz
gunzip $model.gz
mv $model models/
