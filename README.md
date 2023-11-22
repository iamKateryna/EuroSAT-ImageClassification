# Land Use and Land Cover Classification with EuroSAT
This project focuses on classifying land use and land cover using the EuroSAT dataset, which includes 10 different image classes. The goal is to apply advanced machine learning techniques to effectively categorize satellite imagery. 
## Dataset
The EuroSAT dataset comprises satellite images categorized into 10 classes, with each class containing 2000-3000 images. More details can be found in the [EuroSAT paper](https://www.researchgate.net/publication/319463676_EuroSAT_A_Novel_Dataset_and_Deep_Learning_Benchmark_for_Land_Use_and_Land_Cover_Classification).

## Models and Performance
### ResNet50
- Pretrained on ImageNet-1k.
- Achieved an f1 score of 0.99 and AUPRC of 0.99 on test data.

### Vision Transformer (ViT)
- Pretrained on ImageNet-21k.
- Achieved an f1 score of 0.99 and AUPRC of 0.99.

## Methodology
The models were trained on a 60/20/20 training-validation-test split. An f1 score and AUPRC were chosen as metrics because their performance is representative of imbalanced data. Learning rate optimization played a key role in achieving high performance, with lr=10e-5 ResNet50 achieving almost 0.98 f1 score at first epochs, while with lover learning rates (10e-3, 10e-4), the f1 score hardly reached 0.52.

The authors of the dataset used accuracy for model comparison, and it’s 98.57% on an 80/20 training-test spit.

### Training and Evaluation Scripts
- `train.py`: Training script.
- `evaluate.py`: Evaluation script.
- `main.py`: Data preprocessing, training, and evaluation.
- `eurosat.py` & `transformereurosat.py`: Custom PyTorch Dataset classes.

### Example Commands
- Train ViT: `python -u main.py --model_name vit --lr 0.00001 --num_epochs 20`
- Train ResNet50: `python -u main.py --model_name resnet50 --lr 0.00001 --num_epochs 10`


## Inference and Visualization
The [inference notebook](https://github.com/iamKateryna/EuroSAT-ImageClassification/blob/main/landClassificationWithEuroSAT.ipynb) contains scripts for model inference and visual explanations using [Grad-CAM]( https://arxiv.org/pdf/1610.02391.pdf).

## Checkpoints
Trained model checkpoints are available in the [checkpoints folder](https://github.com/iamKateryna/EuroSAT-ImageClassification/tree/main/checkpoints).
