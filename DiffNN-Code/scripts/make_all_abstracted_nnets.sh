#!/usr/bin/env bash
set -e
mkdir -p build
opt=$1; [ "$opt" = "-v" ] || opt=
for f in nnet/*.nnet; do
  q="compressed_nnets/$(basename "${f%.nnet}")_16bit.nnet"
  [ -f "$q" ] && python3 python/cegar_lockstep.py "$f" "$q" -o build $opt
done
