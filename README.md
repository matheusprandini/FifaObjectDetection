# Fifa Object Detection

Using Tensorflow Object Detection API through Google Colab in order to improve the data representation in FIFA 19. 
In particular, this project explores the free kicks task. In this way, the objects detected are: ball, barrier, goalkeeper and targets.
In order to cope with this task, this project used the SSD MobileNet V2 (available in Tensorflow OD API).

## 1 - Creating Dataset

Step 1: Collecting 150 images containing the desired objects (Split training/test = 85%/15%).

Step 2: Annotating the images (labeling the images) in order to create the xml files using [LabelImg](https://github.com/tzutalin/labelImg).

Installing LabelImg:

```
git clone https://github.com/tzutalin/labelImg

sudo apt-get install pyqt5-dev-tools

sudo pip3 install lxml

make qt5py3

python3 labelImg.py
```
Example:

![Fifa Labeling Example](https://github.com/matheusprandini/FifaObjectDetection/blob/master/ImagesReadme/fifa_labeling_example.png)

## 2 - Training the model

The jupyter notebook [FifaObjectDetection](https://github.com/matheusprandini/FifaObjectDetection/blob/master/Fifa_Object_Detection.ipynb) contains all the proccess to train the model in Google Colab using GPU.

## 3 - Testing the model (FINAL MODEL)

Example 1:

![Fifa Labeling Example](https://github.com/matheusprandini/FifaObjectDetection/blob/master/ImagesReadme/fifa_testing_example_1.png)

Example 2:

![Fifa Labeling Example](https://github.com/matheusprandini/FifaObjectDetection/blob/master/ImagesReadme/fifa_testing_example_2.png)
