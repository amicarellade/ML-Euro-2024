# Predicting Euro 2024 Match Outcomes
In this project I used three different models to try and predict the outcomes of the group stage matches in the Euro 2024 tournament. Obtaining data from kaggle.com and sofascore.com I was able to create a more well-rounded dataset in order to train and test on. I created the models using scikit-learns SVM, NN, and Random Forrest packages. Furthermore, I demonstrate the output prediction of each model for each game below. 

![MLModelHeatmap](https://github.com/amicarellade/ML-Euro-2024/assets/56127779/d2e360d0-bfd3-4762-be9b-fed1b9a4b760)

## File Breakdown
- sofascore.py - script to webscrape FIFA ranking from sofascore.com
- euro24.py - scipt to process/clean data, create models and visualize outputs
