#!/bin/bash

DataPath='../DataSplit/KGTN'
OutDir='../results/baseline'

lr=0.1
wd=0.0001

cd KGTN

# Low-shot benchmark without generation
for j in 1 2
do
  for i in {1..5}
  do
    if [ $j == 1 ];then
    maxiters=2000
    elif [ $j == 2 ];then
    maxiters=3000
    else
    maxiters=10000
    fi

    CUDA_VISIBLE_DEVICES=$1 python main_KGTN.py \
      --lowshotmeta ${DataPath}/label_idx.json \
      --experimentpath ${DataPath}/experiment_cfgs/splitfile_{:d}.json \
      --experimentid  $i --lowshotn $j \
      --trainfile ../features/ResNet50_sgm/train.hdf5 \
      --testfile ../features/ResNet50_sgm/val.hdf5 \
      --outdir ${OutDir} \
      --lr ${lr} --wd ${wd} \
      --testsetup 1 \
      --l2_reg 0.001 \
      --maxiters $maxiters 
  done
done

# parse results
echo "baseline"
python ../parse_results.py --resultsdir ${OutDir} \
  --repr ResNet50_sgm \
  --lr ${lr} --wd ${wd}






