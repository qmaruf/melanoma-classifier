# melanoma-classifier

This repository contains a binary classifier to identify melanoma in images of skin lesion. The dataset is collected from Kaggle. This project is inspired by [Abhishek Thakur](https://youtu.be/BUh76-xD5qU), and shows how to build a melanoma classifier and deploy the model through web application.

# steps

First clone the repository using `git clone https://github.com/qmaruf/melanoma-classifier.git`. Download images from this [Kaggle notebook](https://www.kaggle.com/abhishek/melanoma-detection-with-pytorch/data?select=train) and keep them into `./input/224x224/` folder. These images are already resized into 224x224 pixel. Download the `train.csv` file and keep it into the `./input/` folder.

In the `./src/` directory run `python create_folds.py`. It will split the `train.csv` into multiple folds. We will use one fold for evaluation and all other folds to train the classifier. Now run `python main.py --phase train`. You need to install all relevant packages to run this code. After the training, there will be a model with `ckpt` extension in the `./src/` folder. You can use `python main.py --phase test` to evaluate the trained model.

The web app is written use Flask and you can find it inside `../webapp/` folder. Update the `MODEL_PATH` in `./webapp/application.py` using you newly create model. Run the application using `python application.py`.

It should be looked like the following images.
![m1](https://user-images.githubusercontent.com/530250/103720445-14e5b380-5017-11eb-8c91-deb4e49d5884.png | width=10)
![m2](https://user-images.githubusercontent.com/530250/103720457-19aa6780-5017-11eb-9427-41d7480d6e2e.png | width=10)

This classifier is far from perfect and there's lots of scope for improvement. To train and deploy the model quickly, I have only used five epochs.
Happy Classifying.
