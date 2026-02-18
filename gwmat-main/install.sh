#!/bin/bash

## using pip
pip install -e .
rm -r ./build ./*egg-info

## using python steup.py
#python setup.py install
#rm -r ./build ./dist ./*egg-info
