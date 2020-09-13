# Knowledge Graph Transfer Network for Few-Shot Recognition

This is the implementation code of [KGTN paper](https://arxiv.org/pdf/1911.09579.pdf):
  
```bash
@inproceedings{chen2020knowledge,
  title={Knowledge Graph Transfer Network for Few-Shot Recognition.},
  author={Chen, Riquan and Chen, Tianshui and Hui, Xiaolu and Wu, Hefeng and Li, Guanbin and Lin, Liang},
  booktitle={AAAI},
  pages={10575--10582},
  year={2020}
}
```

## Preparation

- Python==2.7
- Pytorch==1.0.1
- Torchvision==0.2.1
  
## Running the code

  Our trained feature extractor is available at [onedrive](https://1drv.ms/u/s!AmTj6SFYjpDHap5ny5DlqGNc2bE?e=sdqen0). *Download the feature extractor and place it into directory `checkpoints/ResNet50_sgm/`*. If do so, go to Step-2.

### 1.Train feature extractor

```bash
./scripts/TrainFeatureExtractor.sh 
```

### 2.  Save feature

```bash
./scripts/SaveFeature.sh
```

### 3. Train the Knowledge Graph Transfer Network (KGTN)

For ***semantic similarity*** knowledge graph:

```bash
./scripts/KGTN_wv_InnerProduct.sh [GPU_ID]
```

or

For ***category hierarchy*** knowledge graph:

```bash
./scripts/KGTN_wordnet.sh [GPU_ID]
```

## Performance

*We take the model with a plain fully-connected layer as the baseline model, which is used in [Low-shot Visual Recognition by Shrinking and Hallucinating Features](https://arxiv.org/abs/1606.02819).

*Top5 Novel*
 Knowledge Graph   | (n-shot) 1 | 2 | 5 |10
-------- | ----- |----- |----- |-----
None(baseline) | 53.7 | 67.5 | 77.8 | 82.2
category hierarchy| 60.3 | 69.9 | 78.3 | 82.3
semantic similarity  | **62.1** | **71.0** | **78.5** | **82.4**

*Top5 All*
 Knowledge Graph   | (n-shot) 1 | 2 | 5 |10
-------- | ----- |----- |----- |-----
None(baseline) |  61.7 | 72.1 | 80.2 | 83.5
category hierarchy| 67.0 | 74.5 | 80.8 | 83.3
semantic similarity  | **68.4** | **75.2** | **80.9** | **83.4**
