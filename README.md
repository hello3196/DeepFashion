# Online Shopping for the Blind

Github for team 42's final project in CS492.

![image](https://user-images.githubusercontent.com/45480548/146496497-b55e2faa-3c6c-433f-9897-d688d97d7d48.png)

# Requirements

Python 3, PyTorch >= 0.4.0, and make sure you have installed TensorboardX:

	pip install tensorboardX

# Quick Start

#### 1. Prepare the Dataset

Download the "Category and Attribute Prediction Benchmark" of the DeepFashion dataset [here](http://mmlab.ie.cuhk.edu.hk/projects/DeepFashion/AttributePrediction.html). Extract all the files to a folder and put all the images in a folder named "img".

Also, for evaluation, we used the images from MUSINSA. Download the images [here](https://drive.google.com/drive/folders/18sg1oCNp5xN4c9CBffdIb01oSqlEuFvd?usp=sharing). The images will be in a folder named "real".

For example, if you choose to put the dataset to /home/user/datasets/benchmark1/, the structure of this folder will be:

	benchmark1/
	    Anno/
	    Eval/
	    img/
	    real/

Please modify the variable "base_path" in src/const.py correspondingly:

	# in src/const.py
	base_path = "/home/user/datasets/benchmark1/"

#### 2. Create info.csv

	python -m src.create_info

Please make sure you have modified the variable "base_path" in src/const.py, otherwise you may encounter a FileNotFound error. After the script finishes, you will find a file named "info.csv" in your "base_path"

#### 3. Train the model

To train the model from scratch, run:

	python -m src.train --conf src.conf.whole

#### 4. Reproduce the results

To reproduce the results for images from MUSINSA, download the pre-trained model "whole.pkl" [here](https://drive.google.com/file/d/1iFLrjPSYTRHPBTtH_723GMe6-BQ5FAK0/view?usp=sharing). The location for the model should be:

	DeepFashion/
	    __MACOSX/
	    models/
	        whole.pkl
	    README.md
	    scripts/
	    src/

To reproduce the results, run:

	python -m src.val --conf src.conf.whole

The original images, attention maps and landmark location maps will be saved as image files in the directory. The output category of each images will be printed in category numbers. You can find out corresponding category in the link of the full dataset above.


