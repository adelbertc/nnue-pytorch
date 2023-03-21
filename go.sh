#!/bin/sh

poetry run python app.py --net data/nn-6877cd24400e.nnue --fens data/fens.txt "${@}"
