#! /usr/bin/env bash
models=(conv smlp lmlp mmlp)

for model in "${models[@]}"; do
    ./tester.py --dataset MNIST --model $model --batchsize 128 --check
    ./tester.py --dataset MNIST --model $model --batchsize 128 --split chaotic
    ./tester.py --dataset MNIST --model $model --batchsize 128 --split uniform
done

models=(conv smlp)
for model in "${models[@]}"; do
    ./tester.py --dataset CIFAR10 --model $model --batchsize 128 --check
    ./tester.py --dataset CIFAR10 --model $model --batchsize 128 --split chaotic
    ./tester.py --dataset CIFAR10 --model $model --batchsize 128 --split uniform
done

mkdir -p resultados
mv -t resultados *.dat
