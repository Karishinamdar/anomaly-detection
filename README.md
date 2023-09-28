### anomaly-detection: Use case- active-wildfire-detection

### Overview
This is the initial layer of onboard anomaly detection on a satellite, with the primary use case being wildfire detection. In the future, this technology can be applied to various other scenarios such as flood detection, maritime vessel detection, deforestation, and more. We are in the process of modyfying this architecture further to gain more efficinecy and acccuracy. 

This repository contains a machine learning model for anomaly detection with use case of detecting active wildfires in satellite imagery using a modified UNet architecture. The model is trained on the Landsat-8 satellite dataset, specifically designed for active wildfire detection. This README file provides an overview of the project, instructions for setting up the environment, training the model, and using it for inference.

# Table of Contents

    Requirements
    Dataset
    Model Architecture
    Training
    Inference
    Evaluation
    Example Usage
    License

### Requirements

To use this project, you'll need the following dependencies:

    Python 3.x
    TensorFlow (or TensorFlow-GPU for GPU acceleration)
    NumPy
    Matplotlib (for visualization)
    OpenCV (for image processing)
    Rasterio (for Geotiff images)
    Jupyter Notebook (for running provided examples)

You can install these dependencies using pip:

    pip install tensorflow numpy matplotlib opencv-python jupyter

### Dataset

The dataset used in this project is the Landsat-8 active wildfire imagery. It contains a collection of satellite images with labels indicating the presence of active wildfires.
To obtain these samples, you have two options:

1. Direct Download: You can directly download the samples from Google Drive.

2. Using Script: Alternatively, you can use the src/utils/download_dataset.py script. Before running the script, set DOWNLOAD_FULL_DATASET to False. Then, execute:

       python download_dataset.py

The samples will be downloaded and stored in the <your-local-repository>/dataset folder. Please note that the output folder differs from the one used for the full dataset to prevent decompression errors when both datasets are present. If you prefer a different download location, you can set the OUTPUT_SAMPLES constant before running the script.

Once you have the samples, you can extract them using the src/utils/unzip_patches.py script. Set FULL_DATASET to False and run the following command:

    python unzip_patches.py

Ensure that you have access to this dataset and organize it as follows:


    dataset/
    ├── train/
    │   ├── images/
    │   │   ├── image1.jpg
    │   │   ├── image2.jpg
    │   │   └── ...
    │   └── masks/
    │       ├── mask1.jpg
    │       ├── mask2.jpg
    │       └── ...
    ├── test/
    │   ├── images/
    │   │   ├── test_image1.jpg
    │   │   ├── test_image2.jpg
    │   │   └── ...
    │   └── masks/
    │       ├── test_mask1.jpg
    │       ├── test_mask2.jpg
    │       └── ...

### Model Architecture

The model architecture used in this project is a modified UNet. UNet is a popular architecture for image segmentation tasks. The modified UNet consists of an encoder-decoder network with skip connections, which helps the model capture both global and local features in the satellite images.

### Training

To train a model from scratch, follow these steps in the src/train/murphy/unet_16f_2conv_762 directory:

Execute the training script by running:

    python train.py

This command will initiate the training process for a new model.

If you have multiple GPUs, you can specify which GPU to use by changing the CUDA_DEVICE constant to the desired GPU number.

Ensure that your dataset is organized as follows:

    Images are located in dataset/images/patches.
    Masks are located in dataset/masks/patches.
    Intersection masks are stored in dataset/masks/intersection.
    Voting masks are stored in dataset/masks/voting.

If your dataset is organized differently, you can modify the IMAGES_PATH and MASKS_PATH constants accordingly.

The training script will produce output files and checkpoints in the train_output folder within the model folder (murphy/unet_16f_2conv_762). Please note that this repository already includes pre-trained weights for the U-Net-Light (3c) models in this folder. If you retrain the model, these weights will be overwritten.

The script will save checkpoints every 5 epochs during training. To resume training from a checkpoint, set the INITIAL_EPOCH constant to the epoch corresponding to the desired checkpoint.

Training may take a while, depending on your hardware and dataset size. You can adjust hyperparameters like batch size, learning rate, and the number of epochs to fine-tune the model's performance.
Inference


### Testing: 

To test the trained models, follow these steps:

Prepare for Testing (Attention: High Memory Usage):
Be cautious as this process loads all data into RAM, potentially causing your machine to freeze in low-memory environments. If needed, reduce the size of images_test.csv and masks_test.csv by removing some rows.

Step 1: CNN Prediction

The first step involves passing the images from images_test.csv through the trained model and saving the output as a txt file. In this txt file, 0 represents the background, and 1 represents fire. Similarly, the masks from masks_test.csv will be converted to a txt file.

  These files will be written to the log folder inside the model folder. The CNN's output predictions will be saved as det_<image-name>.txt, while the corresponding masks will be saved as grd_<mask-name>.txt.

  Execute this process with:

     python inference.py

  You can specify the GPU to use by changing the CUDA_DEVICE constant, and if your samples are in a directory other than the default, modify the IMAGES_PATH and MASKS_PATH constants.

   The CNN's output predictions are converted to integers through a thresholding process. The default threshold is set to 0.25, but you can change this value by modifying the TH_FIRE constant.

Step 2: Model Evaluation

In the second step, you can evaluate your trained model by running:

    python evaluate_v1.py

This script will display the results from your model, providing an assessment of its performance.

### Evaluation

To evaluate the model's performance, you can use standard evaluation metrics such as Intersection over Union (IoU), F1-score, and accuracy. These metrics can help you assess the model's accuracy in detecting active wildfires in satellite imagery.

### CNN Output
To display the CNN output as images, follow these steps:

Access the Script:
        Navigate to the src/utils/cnn folder in your project directory.

  Run the Script:
        Find the generate_inference.py script and open it.

  Configure the Script:
       In the script, configure the following constants:
            IMAGE_NAME: Set this to the desired image name.
            IMAGE_PATH: Specify the path where the image can be found.
            MASK_ALGORITHM: Choose the approach you want to use for mask generation.
            N_CHANNELS and N_FILTERS: Set these to the number of channels and filters used in the model.
            WEIGHTS_FILE: Ensure that the trained weights defined here match the parameters you've set.

Generate the Images:

After configuring the script, execute it with the following command:

    python generate_inference.py

This will pass the specified image patch through the selected model and generate a PNG image with the prediction.

### Output Example:

In src/utils/cnn/output folder you will find png image with the predications 


