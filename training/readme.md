# Training the model

To train the model we first need to prepare the data and then train it with Keras.
The training script expects all the images to be the same size, in our case 455 x 700. 

All the original files need to be in a folder called dataset_ori (or the one you set in the 'input_folder' variable on line 8). If you want the dataset message me, it is about 10.5GB. Also create an empty folder called dataset. Run the prepare python script with `python prepare.py`, you should be seeing images apear in the dataset folder. Depending on the dataset size this may take a while.

Make sure the lines in images.csv match the images in the input_folder. In this file you will say the category the image falls under, for example 'fake'.

## Anaconda

To make it easy to start training you can install [Anaconda](https://www.anaconda.com/distribution/). After you installed this run the following line in a command line: 

`conda create --name tf_gpu tensorflow-gpu`

After this change directory to the training folder and type:

`activate tf_gpu`

Your command line will now begin with `(tf_gpu)`

Install Keras, OpenCV and Pandas with `pip install keras pandas opencv-python`. These will only work in the tf_gpu environment.

If you do not have a supported Nvidia GPU the training will take very long. With my GTX 1060 a dataset of about 12000 images took about 2 hours.

If you have more memory on your GPU you can increase the `batch_size` in the train script for increased performance. If you want you can change the name of the model on line 104 and 122.

Start the train script with `python train.py`. The script will run 70 epochs and it will save models that improve the `val_acc` metric. With my current dataset the `val_acc` is about 99.5%!

To see cool graphs while the model is training you can use Tensorboard. Open a different command line and type `tensorboard --logdir logs`. When started this will show a URL which will open Tensorboard.