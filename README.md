# Land Use and Land Cover Classification task
For this task, I used the whole EuroSAT dataset: 10 image classes with 2000-3000 images per class.

I used two models: ResNet50 - the one that showed the best results in the [paper where the EuroSAT dataset was introduced](https://www.researchgate.net/publication/319463676_EuroSAT_A_Novel_Dataset_and_Deep_Learning_Benchmark_for_Land_Use_and_Land_Cover_Classification).

This model was pretrained on the ImageNet-1k dataset. The f1 and AUPRC scores achieved on test data are '0.9882583022117615' and 0.9987589120864868, respectively. 

I chose these metrics because their performance is representative of imbalanced data. I split the data in a 60/20/20 training-validation-test split, and even though the model was trained on fewer data, the results were pretty good. An important role played learning rate, with lr=10e-5 ResNet50 achieved almost 0.98 f1 score at first epochs, while with lover learning rates (10e-3, 10e-4), the f1 score hardly reached 0.52.

The authors of the dataset used an accuracy metric for model comparison, and it’s 98.57% on an 80/20 training-test spit.

Another model that I used for this task is Vision Transformer. It was pretrained ImageNet-21k and has achieved state-of-the-art performance on a number of image classification benchmarks and has shown great potential for generalizing to new domains.

It showed an even better performance. The f1 and AUPRC scores achieved on test data are 0.9913133382797241 and 0.9989550709724426, respectively. The situation with the learning rate was quite similar to ResNet50: lr=10e-5 showed significantly better results than smaller ones and a bit better than lr=10e-6.

Training script is in .py files:

[train.py](https://github.com/iamKateryna/EuroSAT-ImageClassification/blob/main/train.py) - training

[evaluate.py](https://github.com/iamKateryna/EuroSAT-ImageClassification/blob/main/main.py) - evaluation

[main.py](https://github.com/iamKateryna/EuroSAT-ImageClassification/blob/main/train.py) - data preprocessing and script for training & evaluation

[eurosat.py](https://github.com/iamKateryna/EuroSAT-ImageClassification/blob/main/eurosat.py) and [transformereurosat.py](https://github.com/iamKateryna/EuroSAT-ImageClassification/blob/main/transformereurosat.py) -- custom Dataset classes.

Train the ViT model: python -u main.py --model_name vit --lr 0.00001 --num_epochs 20
Train the ResNet50 model: python -u main.py --model_name resnet50 --lr 0.00001 --num_epochs 10

There is also an Inference script in the [notebook](https://github.com/iamKateryna/EuroSAT-ImageClassification/blob/main/landClassificationWithEuroSAT.ipynb), combined with explainability visualizations (using [Grad-CAM]( https://arxiv.org/pdf/1610.02391.pdf)) - ‘visual explanations’ for decisions from models.

P.S. I like how different the visualisations look for the models

[checkpoints](https://github.com/iamKateryna/EuroSAT-ImageClassification/tree/main/checkpoints) folder contains trained models

