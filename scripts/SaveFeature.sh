mkdir -p features/ResNet50_sgm
cd FeatureExtractor

DataPath='../DataSplit/FeatureExtractor'

ModelFile='../checkpoints/ResNet50_sgm/89.tar'
TrainOutFile='../features/ResNet50_sgm/train.hdf5'
ValOutFile='../features/ResNet50_sgm/val.hdf5'

echo "Saving train feature..."
CUDA_VISIBLE_DEVICES=5 python ./save_features.py \
      --cfg ${DataPath}/train_save_data.yaml \
      --outfile ${TrainOutFile} \
      --modelfile ${ModelFile} \
      --model ResNet50
echo "Finishing"

echo "Saving val feature..."
CUDA_VISIBLE_DEVICES=5 python ./save_features.py \
      --cfg ${DataPath}/val_save_data.yaml \
      --outfile ${ValOutFile} \
      --modelfile ${ModelFile} \
      --model ResNet50
echo "Finishing
