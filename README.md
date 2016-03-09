# deep-features

Creating a machine learning model to recognize Korean foods in photos using Google's Inception-v3 Convolutional Neural Network to generate and analyze deep features.

The resulting TensorFlow model was used to develop a [Django web server](https://github.com/jjinking/kfood-server) and an [Android app](https://github.com/jjinking/kfood-android) that can send photos of Korean food to the web server and receives back the classification label.

## Requirements

- TensorFlow v6.0
- [Datsci](https://github.com/jjinking/datsci/) - my personal library used for data science-related projects
- jupyter==1.0.0
- matplotlib==1.5.1
- MetaMindApi==1.2.2
- numpy==1.10.4
- pandas==0.17.1
- Pillow==3.1.0
- prettytable==0.7.2
- scikit-learn==0.17.1
- scipy==0.17.0
- seaborn==0.7.0


## Machine Learning

[Full report](https://github.com/jjinking/deep-features/blob/master/report.pdf)

3,470 photos of 20 Korean dishes were used for this project.

| Name           |  Number of samples
| -----------		 | ------------------
| galbijjim      |  219
| kimbab         |  217
| bibimbab       |  214
| hotteok        |  210
| nangmyun       |  207
| dakgalbi       |  205
| sullungtang    |  193
| japchae        |  193
| bulgogi        |  183
| samgyupsal     |  173
| bossam         |  173
| dakbokeumtang  |  171
| jeyookbokkeum  |  165
| samgyetang     |  158
| ddukbokee      |  150
| lagalbi        |  148
| jeon           |  134
| kimchi         |  127
| ramen          |  118
| yookgyejang    |  112


For transfer learning, TensorFlow was used to retrain the Inception-v3 network's final layer.

![Training Accuracy \label{figure_train_acc}](report_images/train_acc.png "TensorFlow Training Accuracy")

![Validation Accuracy \label{figure_valid_acc}](report_images/valid_acc.png "Validation Accuracy During Training")

![Cross Entropy \label{figure_cross_ent}](report_images/cross_ent.png "Cross Entropy During Training")

![Confusion Matrix on Test Set \label{figure_cm}](report_images/retrained_cm.png "Confusion Matrix on Test Set")
