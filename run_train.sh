#!/bin/bash
echo running

for (( c=1; c<=10; c++ ))
do
        python train_oldfashion.py
done
