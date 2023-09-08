%env SM_FRAMEWORK=tf.keras
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import keras
import cv2
import segmentation_models as sm
from IPython.display import Image, display
from tensorflow.keras.preprocessing.image import load_img
import PIL
from PIL import ImageOps
from skimage import color
from segmentation_models.losses  import bce_jaccard_loss
from segmentation_models.metrics import IOUScore
from tqdm import tqdm
import keras.backend as K



# auxiliary utility functions
def extract_mask_from_ground(ground_img, lower_color_threshold):
    (R, G, B) = cv2.split(ground_img)
    
    # Values are clipped as it is a binary classification problem
    R_ranged = np.clip(cv2.inRange(R, lower_color_threshold, 255), 0, 1) 
    G_ranged = np.clip(cv2.inRange(G, lower_color_threshold, 255), 0, 1) 
    B_ranged = np.clip(cv2.inRange(B, lower_color_threshold, 255), 0, 1)

    merged = cv2.merge([B_ranged, G_ranged, R_ranged])
    merged = np.expand_dims(merged, 3)
    merged = np.amax(merged, axis=2)
    return merged

# The threshold value determines how many small rocks are visible, higher threshold means less blue rocks
def load_ground(path, color_threshold):
    img_ground = cv2.imread(path)
    img_ground_rgb = cv2.cvtColor(img_ground, cv2.COLOR_BGR2RGB)
    img_ground_rgb = cv2.resize(img_ground_rgb, img_size[::-1])
    return extract_mask_from_ground(img_ground_rgb, color_threshold)

class ResyrisRocks(keras.utils.Sequence):
    def __init__(self, batch_size, img_size, input_img_paths, target_img_paths, color_threshold):
        self.batch_size = batch_size
        self.img_size = img_size
        self.input_img_paths = input_img_paths
        self.target_img_paths = target_img_paths
        self.color_threshold = color_threshold
        
    def __len__(self):
        return len(self.target_img_paths) // self.batch_size
    
    def __getitem__(self, idx):
        # Returns tuple (input, target) corresponding to batch idx
        i = idx * self.batch_size
        batch_input_img_paths = self.input_img_paths[i : i + self.batch_size]
        batch_target_img_paths = self.target_img_paths[i : i + self.batch_size]
        
        # 3 Dimensional input images
        x = np.zeros((self.batch_size,) + self.img_size + (3,), dtype="float32")
        for j, path in enumerate(batch_input_img_paths):
            img = load_img(path, target_size = self.img_size)
            x[j] = img
                
        # 1 Dimensional target mask
        y = np.zeros((self.batch_size,) + self.img_size + (1,), dtype="uint8")
        for j, path in enumerate(batch_target_img_paths):
            # load ground imahe
            y[j] = load_ground(path, self.color_threshold)
                
        return x, y

def compressed_predict(interpreter, input_paths):
    input_index = interpreter.get_input_details()[0]["index"]
    output_index = interpreter.get_output_details()[0]["index"]
    # Run predictions on ever y image in the "test" dataset.
    predictions = []
    for i, test_image_path in tqdm(enumerate(input_paths)):
        # if i % 100 == 0:
            # print(i)
        # Pre-processing: add batch dimension and convert to float32 to match with
        # the model's input data format.
        test_image = load_img(test_image_path, target_size = img_size)
        # print(type(test_image))
        # test_image = test_image.astype(np.float32)
        test_image = np.expand_dims(test_image, axis=0).astype(np.float32)
        # print(test_image.shape)
        interpreter.set_tensor(input_index, test_image)

        # Run inference.
        interpreter.invoke()

        # Post-processing: remove batch dimension and find the digit with highest
        # probability.
        pred = interpreter.get_tensor(output_index)
        pred = pred.squeeze(axis=0)
        predictions.append(pred)


    return predictions

# Calculate dice coefficent for one input array
def dice_coef(y_true, y_pred):
    y_true_f = y_true.flatten()
    y_pred_f = y_pred.flatten()
    intersection = np.sum(y_true_f * y_pred_f)
    smooth = 0.0001
    return (2. * intersection + smooth) / (np.sum(y_true_f) + np.sum(y_pred_f) + smooth)

# Calculate average dice loss for every label
def dice_coef_multilabel(y_true, y_pred, numLabels):
    dice = [0, 0]
    result = 0
    average = 0
    for index in range(numLabels):
        result = dice_coef(y_true[:,:,index], y_pred[:,:,index]) # @Todo: Zbierać dice score dla każdej klasy
        # print("Dice score for class {}: {}".format(index, result))
        dice[index] += result
        average += result
    return (dice, average/numLabels) # taking average

#@todo: make function for evaluating dice loss
def calculate_dice_loss(preds, target_paths, color_threshold=1):
    result = [0,0]
    average = []
    for i, mask in enumerate(preds):
        mask = np.argmax(mask, axis=2)
        mask = np.expand_dims(mask, axis=-1)
        mask = mask - 1
        mask = np.clip(mask, 0, 1)
        target = cv2.imread(target_paths[i])
        target = cv2.cvtColor(target, cv2.COLOR_BGR2RGB)

        target = cv2.resize(target, img_size[::-1])
        target = extract_mask_from_ground(target, color_threshold)

      #One hot encoding so the Dice coefficent works
        mask = tf.one_hot(mask, 2)
        mask = np.squeeze(mask, axis=2)
        target = tf.one_hot(target, 2)
        target = np.squeeze(target, axis=2)
#         print(mask.shape)
#         print(target.shape)
        dice_coef_tuple = dice_coef_multilabel(target, mask, 2)
        average.append(dice_coef_tuple[1])
        for i, value in enumerate(dice_coef_tuple[0]):
            result[i] += value

    for i, value in enumerate(result):
        print("Dice score for class {}: {}".format(i, value/len(preds)))
    return average

# Calculate metrics for all model predictions, returns a list 
def evaluate_model(preds, target_paths_here, metric, color_threshold=1):
    result = []
    metric.reset_states()
    for i, mask in enumerate(preds):
        mask = np.argmax(mask, axis=2)
        mask = np.expand_dims(mask, axis=-1)
        mask = mask - 1
        mask = np.clip(mask, 0, 1)
#         print(np.unique(mask))
        target = cv2.imread(target_paths_here[i])
        target = cv2.cvtColor(target, cv2.COLOR_BGR2RGB)

        target = cv2.resize(target, img_size[::-1])
        target = extract_mask_from_ground(target, color_threshold)

        metric.update_state(target, mask)
        result.append(metric.result().numpy())
    return result

def eval_preds(preds, img_paths, metric_name="", color_threshold = 1):
    # Define metrics
    meanIoU = tf.keras.metrics.MeanIoU(num_classes=2)
    accuracy = tf.keras.metrics.Accuracy()
    rmse = tf.keras.metrics.RootMeanSquaredError()

    # Calculate metrics for validation set
    dice_loss = np.mean(calculate_dice_loss(preds, img_paths, color_threshold))
    iou = np.mean(evaluate_model(preds, img_paths, meanIoU, color_threshold))
    accuracy =  np.mean(evaluate_model(preds, img_paths, accuracy, color_threshold))
    rmse = np.mean(evaluate_model(preds, img_paths, rmse, color_threshold))

    metrics = [dice_loss, iou, accuracy, rmse]
    
    # Display metrics
    print("Dice score on {} set: {}".format(metric_name, dice_loss))
    print("Mean IoU on {} set: {}".format(metric_name, iou))
    print("Accuracy on {} set: {}".format(metric_name, accuracy))
    print("RMSE on {} set: {}".format(metric_name,rmse))
    print()

    return metrics


if __name__ == '__main__':
    # Prepare paths of input images and target segmentation masks
    input_dir= "../datasets with noise/Resyris real"
    target_dir= "../resyris/test_data_realworld/gt/"


    img_size = (320,320)
    # img_size = (336, 336)

    input_real_img_paths = sorted([
        os.path.join(input_dir, fname)
        for fname in os.listdir(input_dir)
        if fname.endswith(".png")
    ])

    #Sorting mask names in target folder
    target_real_img_paths = sorted([
        os.path.join(target_dir, fname)
        for fname in os.listdir(target_dir)
        if fname.endswith(".png") and not fname.startswith(".")
    ])


    print("Number of input samples:", len(input_real_img_paths))
    print("Number oftarget samples:", len(target_real_img_paths))

    for input_path, target_path in zip(input_real_img_paths[:10], target_real_img_paths[:10]):
        print(input_path, "|", target_path)

    # Prepare paths of input images and target segmentation masks
    input_dir= "../datasets with noise/Resyris synth"
    target_dir= "../resyris/test_data_synthetic/gt/"


    img_size = (320,320)
    # img_size = (336, 336)

    input_synth_img_paths = sorted([
        os.path.join(input_dir, fname)
        for fname in os.listdir(input_dir)
        if fname.endswith(".png")
    ])

    #Sorting mask names in target folder
    target_synth_img_paths = sorted([
        os.path.join(target_dir, fname)
        for fname in os.listdir(target_dir)
        if fname.endswith(".png") and not fname.startswith(".")
    ])


    print("Number of input samples:", len(input_synth_img_paths))
    print("Number oftarget samples:", len(target_synth_img_paths))

    for input_path, target_path in zip(input_synth_img_paths[:10], target_synth_img_paths[:10]):
        print(input_path, "|", target_path)

        real_data_generator = ResyrisRocks(16, img_size, input_real_img_paths, 
            target_real_img_paths, color_threshold = 1)

    synth_data_generator = ResyrisRocks(16, img_size, input_synth_img_paths, target_synth_img_paths, color_threshold = 1)

# This line downloads models from mlflow artifacts
    model_path = ''
    temp_dir = mlflow.artifacts.download_artifacts()
    print(temp_dir)
    model = keras.models.load_model(temp_dir + '/data/model', compile=False)
    model.summary()

    converter = tf.lite.TFLiteConverter.from_saved_model(temp_dir + '/data/model') # path to the SavedModel directory
    converter.optimizations = [tf.lite.Optimize.DEFAULT]

    # with mlflow.start_run(run_name="LinkNet with MobileNetV2, compressed") as run:
    tflite_model = converter.convert()
    interpreter = tf.lite.Interpreter(model_content=tflite_model)
    interpreter.allocate_tensors()

    real_preds_compressed = compressed_predict(interpreter, input_real_img_paths)
    synth_preds_compressed = compressed_predict(interpreter, input_synth_img_paths)

    real_metrics = eval_preds(real_preds, target_real_img_paths, 'resyris real')

    real_metrics_compressed = eval_preds(real_preds_compressed, target_real_img_paths, 'resyris real compressed')

    synth_metrics = eval_preds(synth_preds, target_synth_img_paths, 'resyris synth')

    synth_metrics_compressed = eval_preds(synth_preds_compressed, target_synth_img_paths, 'resyris synth compressed')