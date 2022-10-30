#! /usr/bin/env bash
models=(smlp lmlp mmlp conv)

for model in "${models[@]}"; do
    ./tester.py --dataset MNIST --model $model --batchsize 128 --check
    ./tester.py --dataset MNIST --model $model --batchsize 128 --split chaotic --combine mean
    ./tester.py --dataset MNIST --model $model --batchsize 128 --split uniform --combine mean
    ./tester.py --dataset MNIST --model $model --batchsize 128 --split chaotic --combine coln
    ./tester.py --dataset MNIST --model $model --batchsize 128 --split uniform --combine coln
done

models=(smlp conv)
for model in "${models[@]}"; do
    ./tester.py --dataset CIFAR10 --model $model --batchsize 128 --check
    ./tester.py --dataset CIFAR10 --model $model --batchsize 128 --split chaotic --combine mean
    ./tester.py --dataset CIFAR10 --model $model --batchsize 128 --split uniform --combine mean 
    ./tester.py --dataset CIFAR10 --model $model --batchsize 128 --split chaotic --combine coln
    ./tester.py --dataset CIFAR10 --model $model --batchsize 128 --split uniform --combine coln 
done

models=(smlp boost)
for model in "${models[@]}"; do
    ./tester.py --dataset WISCONSIN --model $model --batchsize 128 --check
    ./tester.py --dataset WISCONSIN --model $model --batchsize 128 --split chaotic --combine mean --no-gpu
    ./tester.py --dataset WISCONSIN --model $model --batchsize 128 --split uniform --combine mean --no-gpu 
    ./tester.py --dataset WISCONSIN --model $model --batchsize 128 --split chaotic --combine coln --no-gpu
    ./tester.py --dataset WISCONSIN --model $model --batchsize 128 --split uniform --combine coln --no-gpu 
done

mkdir -p resultados
mv -t resultados-corretos *.csv
