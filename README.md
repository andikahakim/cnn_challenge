# ACSE 4.4 - Machine Learning Miniproject
## The Identification Game Overview
The purpose of this project is to train an image classifier, all information (including the data) is available here: https://www.kaggle.com/c/acse-miniproject/overview

## Getting started 
### Prerequisites
```Python``` and the libraries found in ```requirements.txt``` are prerequisites.

### Installing
To clone the repository to your local machine enter the following in your terminal:
```
git clone https://github.com/acse-2019/acse4-4-dropout.git
```
Due to the size of our models they are stored here: https://drive.google.com/open?id=1l1D7Jp4zcP1cdaev9sUBJbpuqfBlSeFM. We provide our best model from each day: 

```best_resnet101wide_21_02_2020.pth``` (Thursday, best), <br /> 
```resnet101wide_20_02_2020.pth``` (Wednesday, 2nd best), <br /> 
```resnet18_dropout_19_02_2020.pth``` (Tuesday, 3rd best). 

### File guide
```single_model.ipynb``` code to run our single model, both to train and/or to load our model and test it. This notebook takes long to run due to training the models, so we have submitted it with the cells cleared <br /> 
```ensembler_model.ipynb``` code to run our ensemble model (i.e. combination of models) <br /> 
```data_analysis.ipynb``` code to explore the dataset and generate our data analysis charts model <br /> 
```web_scraper.ipynb``` code for our experimental web scraping tool which grabs more ImageNet photos. The notebook has not been run as running it produces no output. The only side effect of the code is the images it generates from urls <br /> 
```model_comparisons.xlsx``` spreadsheet comparing some of the models run during the week <br /> 
```requirements.txt``` contains required libraries <br /> 
```presentation_final.pptx``` contains the slides for our presentation

## User Instructions
#### Loading the data
Data can be downloaded from the Kaggle link provided above, use the following code to instantiate a DataLoader. This is a custom DataLoader which we wrote, more details can be found in the notebooks:
```
test_data_raw = ImageFolderWithPaths(path+'test', transform=test_transform)
test_loader = DataLoader(test_data_raw, batch_size=1, shuffle=False, num_workers=0)
```
, where ```path``` is set as the path to the directory you have stored the data in. 


#### Loading a model
First ensure you have downloaded the desired model from the GoogleDrive link above and have it in the same directory as the Jupyter Notebook, also ensure the specified libraries in our Notebook have been installed. Below, we show how to download the model parameters for our best single model in our ```single_model.ipynb``` notebook:
```
resnetmodel = choose_model("wide_resnet101_2", printmod=True)
resnetmodel.load_state_dict(torch.load(F"../best_resnet101wide_21_02_2020.pth"))
```
, note that ```torch.nn.functional``` was imported as ```F```. You are now ready to use the model to make predictions from input images.
#### Making & saving the model predictions
The following commands will generate the required csv file, which contains label predictions for each input image:
```
y_preds, filenames = test(resnetmodel, test_loader)
submission = pd.DataFrame({'Filename': filenames, 'Label': y_preds})
submission.head()
submission.to_csv('wide_resnet101_2.csv', index=False)
```


## Workflow: 
#### Pre-processing
We were provided with a training set of 100,000 jpeg images with 200 different class labels. Most images have 3 channels for colour, RGB, so are 64x64x3 arrays. The data was downloaded using ImageFolder. A transform was applied at this point to: (1) resize images to the required input size for each network; (2) apply channel-wise normalisation; and later (3) apply data augmentation strategies such as horizontal flipping or rotation. 
We then split up the data into training and validation datasplits (80/20), note that when training our final model we used all of the available data. Finally, we instantiated DataLoaders. Some example images with their associated label are shown below (Figure 1).

![Figure 1](https://github.com/acse-2019/acse4-4-dropout/blob/master/figure1.PNG)
#### Figure 1: Examples of some training data.
#### Machine Learning Approaches
We investigated several different approaches during the course of this project: (1) producing our own CNN from scratch; (2) transfer learning; and (3) Ensemble methods. We found transfer learning using models pre-trained on ImageNet and then fine-tuning them to be highly effective. Despite the risk of overtraining we found that increasing both the width and depth of our CNN resulted in greater accuracies. Due to time constraints, we could not fully investigate the viability of Ensemble methods, but the initial results are promising. 


#### Training & Hyper-parameter Optimisation
We used a simple grid search to optimise our hyperparameters, where we changed one hyperparameter over a specified range and kept all other parameters constant. The results for the wide Resnet 101 layer model are shown below in Table 1. We also tried different optimisation strategies such as RMSProp or using Adam, but found that stochastic gradient descent with momentum produced optimal results. 
We selected the best network based and set of hyperparameters based off our validation accuracies, which were obtained using the sci-kit learn accuracy_score function. We then re-ran our optimal model using the whole dataset. The results from some of our best models are shown further below. 


#### Results
Below we provide some example results from running different models and/or hyperparameters (Table 1), as well as a plot of validation/training accuracy for our best model (Figure 2). We note that in general, deeper and wider models result in a higher accuracy, with our best performing model being the resnet101 wide configuration (```wide_resnet101_2```).
|          Model         |                  |             |                    | Hyper Parameters |               |           |               |          |              |                  |          |   Result   |                   |                     |
|:----------------------:|:----------------:|:-----------:|:------------------:|:----------------:|:-------------:|:---------:|:-------------:|:--------:|:------------:|:----------------:|:--------:|:----------:|:-----------------:|:-------------------:|
|        *pth_name       |       Name       | Fine Tuning | Feature Extraction |    Activation    | Augmentation* | Optimizer | Learning Rate | Momentum | Weight Decay | Train Batch size | Epochs** | Proportion | Training Accuracy | Validation Accuracy |
|     resnet18_1.pth     |     resnet18     |      v      |          -         |       ReLU       |       -       |    SGD    |     1E-02     |    0.5   |       0      |        500       |     3    |   80 : 20  | 70.9%             | 76.5%               |
|    densenet121_1.pth   |    densenet121   |      v      |          -         |       ReLU       |       -       |    SGD    |     1E-02     |    0.5   |       0      |        64        |     3    |   70 : 30  | 88.1%             | 73.5%               |
|  resnext50_32x4d_1.pth |  resnext50_32x4d |      v      |          -         |       ReLU       |       -       |    SGD    |     1E-02     |    0.5   |       0      |        100       |     3    |   70 : 30  | 90.0%             | 80.7%               |
|     resnet152_1.pth    |     resnet152    |      v      |          -         |       ReLU       |       -       |    SGD    |     1E-02     |    0.5   |     1E-05    |        100       |     3    |   70 : 30  | 80.9%             | 80.9%               |
| wide_resnet101_2_1.pth | wide_resnet101_2 |      v      |          -         |       ReLU       |       -       |    SGD    |     1E-02     |    0.6   |       0      |        64        |     2    |   70 : 30  | 88.1%             | 81.9%               |
| wide_resnet101_2_9.pth | wide_resnet101_2 |      v      |          -         |       ReLU       | Rota10, Hflip |    SGD    |     1E-03     |    0.6   |     1E-05    |        64        |     6    |   80 : 20  | 90.4%             | 84.7%               |

\* Aside from normalization with mean and std 
\** Stopped after overfit occured between validation and training accuracy 

#### Table 1: Examples of some of our results, models are pre-trained on ImageNet then fine-tuned. More results can be found in ```model_comparisons.xlsx```.
![Figure 2](https://github.com/acse-2019/acse4-4-dropout/blob/master/figure2.png)
#### Figure 2: A plot of validation and training accuracy against number of epochs for our best model (best_resnet101wide_21_02_2020.pth) with our optimal hyperparameters (given in our code). 

#### Conclusion
We found that the highest accuracies were obtained using a transfer learning approach, where we used a model pre-trained on ImageNet data and fine-tuned the whole model. Additionally, wider and deeper CNNs tended to produce better results despite the risk of over-fitting. Our best single-model result was achieved using the wide resnet101.

An Ensemble approach provides an opportunity to further improve results by incorporating several models, however due to time constraints we could not fully investigate this. Our Ensemble approach used a weighted softmax voting algorithm to make class predictions. The weights were based on Kaggle scores or validation test accuracy due to a lack of unseen data which meant we could not train the Ensemble to learn the optimal weights. Due to this constraint we found that we slightly overfit the public leaderboard dataset, hence our performance dropped from second to fourth place on the private leaderboard (which had new data). Having per class weights, per model would have helped us mitigate this problem and is something we would implement going forward. Ideally, we would also more robustly optimise our hyperparameters, for example by using a Bayesian approach. 
