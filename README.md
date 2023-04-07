# Land Use and Land Cover Classification task
For this task, I used the whole EuroSAT dataset: 10 image classes with 2000-3000 images per class.

I used two models. The first one is ResNet50 - the one that showed the best results in the [paper that introduced the EuroSAT dataset](https://www.researchgate.net/publication/319463676_EuroSAT_A_Novel_Dataset_and_Deep_Learning_Benchmark_for_Land_Use_and_Land_Cover_Classification). This model was pretrained on the ImageNet-1k dataset. The f1 and AUPRC scores achieved on test data are 0.988 and 0.999, respectively. 

I chose these metrics because their performance is representative of imbalanced data. The data was split in a 60/20/20 training-validation-test splits, and even though the model was trained on fewer data, than in original paper, the results were equally good. Learning rate played an important role in training, with lr=10e-5 ResNet50 achieved almost 0.98 f1 score at first epochs, while with lover learning rates (10e-3, 10e-4), the f1 score hardly reached 0.52.

The authors of the dataset used accuracy for model comparison, and it’s 98.57% on an 80/20 training-test spit.

The second model is Vision Transformer. It was pretrained ImageNet-21k and has achieved state-of-the-art performance on a number of image classification benchmarks and has shown great potential for generalizing to new domains.

It showed better performance. The f1 and AUPRC scores achieved on test data are 0.991 and 0.999, respectively. The influence of learning rate was similar to training with ResNet50 model: lr=10e-5 showed significantly better results than smaller ones and a bit better than lr=10e-6.

Training script is in .py files:

[train.py](https://github.com/iamKateryna/EuroSAT-ImageClassification/blob/main/train.py) - training

[evaluate.py](https://github.com/iamKateryna/EuroSAT-ImageClassification/blob/main/main.py) - evaluation

[main.py](https://github.com/iamKateryna/EuroSAT-ImageClassification/blob/main/train.py) - data preprocessing and script for training & evaluation

[eurosat.py](https://github.com/iamKateryna/EuroSAT-ImageClassification/blob/main/eurosat.py) and [transformereurosat.py](https://github.com/iamKateryna/EuroSAT-ImageClassification/blob/main/transformereurosat.py) -- custom pyTorch Dataset classes.

Train the ViT model: python -u main.py --model_name vit --lr 0.00001 --num_epochs 20
Train the ResNet50 model: python -u main.py --model_name resnet50 --lr 0.00001 --num_epochs 10

There [notebook](https://github.com/iamKateryna/EuroSAT-ImageClassification/blob/main/landClassificationWithEuroSAT.ipynb), containes an inference script with explainability visualizations (using [Grad-CAM]( https://arxiv.org/pdf/1610.02391.pdf)) - ‘visual explanations’ for decisions from models.

[checkpoints](https://github.com/iamKateryna/EuroSAT-ImageClassification/tree/main/checkpoints) folder contains trained models
