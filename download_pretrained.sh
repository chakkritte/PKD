#!/bin/sh

mkdir pre-trained

wget https://github.com/chakkritte/PKD/releases/download/v1/model_ofa1k.pt -O pre-trained/model_ofa1k.pt

wget https://github.com/chakkritte/PKD/releases/download/v1/model_efb4.pt -O pre-trained/model_efb4.pt

wget https://github.com/chakkritte/PKD/releases/download/v1/model_pnasnet5_1k.pt -O pre-trained/model_pnasnet5_1k.pt

echo "Download salicon pretrained model to pre-trained directory"