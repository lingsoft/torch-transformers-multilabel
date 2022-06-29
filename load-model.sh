#!/bin/bash

# Load trained models
mkdir -p models
#model=xlmr-base-fi-se.pt
model=xlmr-base-en-fi-fr-se.pt
wget http://dl.turkunlp.org/register-labeling-model/$model.gz
gunzip $model.gz
mv $model models/
