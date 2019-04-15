# Diseases-Detection-on-Chest-X-Rays

Diseases (14 diseases labels) Detection on Chest X-Rays image with Deep Learning (Convolutional Neural Network).



# training:
I randomly split the dataset into training (70%), validation (10%), and test (20%) sets (same [CheXNet](https://arxiv.org/abs/1711.05225)), but I use an initial learning rate of 0.0001(and the [CheXNet](https://arxiv.org/abs/1711.05225) use 0.001) that is decayed by a factor of 3 each time (for fast result) the validation loss plateaus after an epoch, and pick the model with the lowest validation loss.
I use inception-resnet-v2 instead of densnet121 ([CheXNet](https://arxiv.org/abs/1711.05225) model). Also not to overfit I use 3-way image augmentation:
1. random flip 
2. random crop 
3. random rotate 

# result:
I have also proposed a slightly-improved model which achieves a mean AUROC of **0.852**(v.s. 0.841 of the original [CheXNet](https://arxiv.org/abs/1711.05225)).

| Pathology | My Model | [CheXNet](https://arxiv.org/abs/1711.05225) | 
| ------------- | ------------- | ------------- | 
| Atelectasis  | 0.8296  | 0.8094 |
| Cardiomegaly  | 0.9046  | 0.9248 |
| Effusion  | 0.8850  | 0.8638 |
| Infiltration  | 0.7053  | 0.7345 |
| Mass  | 0.8729  | 0.8676 |
| Nodule | 0.8039 | 0.7802 |
| Pneumonia | 0.7876 | 0.7680 |
| Pneumothorax | 0.8920 | 0.8887 |
| Consolidation | 0.8299 | 0.7901 |
| Edema | 0.9171 | 0.8878 |
| Emphysema | 0.9131 | 0.9371 |
| Fibrosis | 0.8402 | 0.8047 |
| Pleural_Thickening | 0.8174 | 0.8062 |
| Hernia | 0.9370 | 0.9164 |

## Requirement

- Python 3.6+
- Keras
- Pandas
- skimage

## GPU and Computation time
The training was done using single 1080 Ti GPU and took approximately 12h.








