# This var is needed by sm models to work
%env SM_FRAMEWORK=tf.keras

# Basic necessary imports
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import keras
import cv2
import imgaug as ia
import imgaug.augmenters as iaa
from imgaug.augmentables.segmaps import SegmentationMapsOnImage
import segmentation_models as sm
from skimage import color
from IPython.display import Image, display
from tensorflow.keras.preprocessing.image import load_img
import PIL
from PIL import ImageOps
from keras import layers
from  mlflow.tracking import MlflowClient
import random
import keras.backend as K


def extract_mask_from_ground(ground_img, lower_color_threshold):
  kernel = np.ones((10,10), np.uint8)

  (R, G, B) = cv2.split(ground_img)
  R_ranged = np.clip(cv2.inRange(R, 150, 255), 0, 1)  # Value for red is hard coded as to not get problems with sky color clipping
  G_ranged = np.clip(cv2.inRange(G, lower_color_threshold, 255), 0, 2)
  B_ranged = np.clip(cv2.inRange(B, lower_color_threshold, 255), 0, 3)

  # B_ranged = cv2.morphologyEx(B_ranged, cv2.MORPH_CLOSE, kernel)  #Clipping does a good job with holes in big rocks, not needed here
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

    class LunarLandscape(keras.utils.Sequence):
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

class LunarLandscapeAgumentation(LunarLandscape):
    def __getitem__(self, idx):
        # Returns tuple (input, target) corresponding to batch idx
        i = idx * self.batch_size
        batch_input_img_paths = self.input_img_paths[i : i + self.batch_size]
        batch_target_img_paths = self.target_img_paths[i : i + self.batch_size]

        # Define agumentation pipeline so the same transformation is applied on both input and mask
        seq = iaa.Sequential([
            iaa.Fliplr(0.5), # horizontal flips
            iaa.Flipud(0.5), # vertically flip 50% of all images
            iaa.Crop(percent=(0, 0.1)), # random crops
            iaa.LinearContrast((0.75, 1.5)),
        ], random_order=True)

        # 3 Dimensional input images
        x = np.zeros((self.batch_size,) + self.img_size + (3,), dtype="float32")
        # 1 Dimensional target masks
        y = np.zeros((self.batch_size,) + self.img_size + (4,), dtype="float32")  #Also here the depth is changed
        for j, path in enumerate(batch_input_img_paths):
            img = load_img(path, target_size = self.img_size)
            mask = load_ground(batch_target_img_paths[j], self.color_threshold)

            img, mask = seq(image=np.asarray(img), segmentation_maps=[mask])

            x[j] = img.astype(np.float32)

            # These two lines only for Dice loss with sm library

            # mask = tf.one_hot(mask[0], 4)
            # mask = np.squeeze(mask, axis=2)
            # print(len(mask))
            y[j] = mask[0].astype(np.float32)
            # y[j] = mask.astype(np.float32)

        return x, y

def get_unet_model(img_size, num_classes):
  model = sm.Unet(backbone_name='mobilenet', input_shape=img_size + (3, ), classes=num_classes)
  return model

def get_linknet_model(img_size, num_classes):
  model = sm.Linknet(backbone_name='mobilenetv2', input_shape= img_size + (3, ), classes=num_classes)
  return model

def get_FPN_model(img_size, num_classes):
  model = sm.FPN(backbone_name='mobilenet', input_shape = img_size + (3, ), classes=num_classes)
  return model

def get_PSPNet_model(img_size, num_classes):
  model = sm.PSPNet(backbone_name='mobilenet', input_shape = img_size + (3, ), classes=num_classes, downsample_factor=8)
  return model

  def compressed_predict(interpreter, input_paths):
  input_index = interpreter.get_input_details()[0]["index"]
  output_index = interpreter.get_output_details()[0]["index"]
  # Run predictions on ever y image in the "test" dataset.
  predictions = []
  for i, test_image_path in enumerate(input_paths):

    if i % 100 == 0:
      print(i)
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

def get_mask(i, preds):
 #Quick utilization to display a model's prediction

    mask = np.argmax(preds[i], axis=2)
    mask = np.expand_dims(mask, axis=-1)
    # img = PIL.ImageOps.autocontrast(tf.keras.preprocessing.image.array_to_img(mask))
    return mask

# Calculate dice coefficent for one input array
def dice_coef(y_true, y_pred):
    y_true_f = y_true.flatten()
    y_pred_f = y_pred.flatten()
    intersection = np.sum(y_true_f * y_pred_f)
    smooth = 0.0001
    return (2. * intersection + smooth) / (np.sum(y_true_f) + np.sum(y_pred_f) + smooth)

# Calculate average dice loss for every label
def dice_coef_multilabel(y_true, y_pred, numLabels):
    dice = [0, 0, 0, 0]
    result = 0
    average = 0
    for index in range(numLabels):
        result = dice_coef(y_true[:,:,index], y_pred[:,:,index]) # @Todo: Zbierać dice score dla każdej klasy
        # print("Dice score for class {}: {}".format(index, result))
        dice[index] += result
        average += result
    return (dice, average/numLabels) # taking average


def calculate_dice_loss(preds, target_paths, color_threshold):
  result = [0,0,0,0]
  average = []
  for i, mask in enumerate(preds):
      mask = np.argmax(mask, axis=2)
      mask = np.expand_dims(mask, axis=-1)
      target = cv2.imread(target_paths[i])
      target = cv2.cvtColor(target, cv2.COLOR_BGR2RGB)

      target = cv2.resize(target, img_size[::-1])
      target = extract_mask_from_ground(target, color_threshold)

      #One hot encoding so the Dice coefficent works
      mask = tf.one_hot(mask, 4)
      mask = np.squeeze(mask, axis=2)
      target = tf.one_hot(target, 4)
      target = np.squeeze(target, axis=2)

      dice_coef_tuple = dice_coef_multilabel(target, mask, 4)
      average.append(dice_coef_tuple[1])
      for i, value in enumerate(dice_coef_tuple[0]):
        result[i] += value

  for i, value in enumerate(result):
    print("Dice score for class {}: {}".format(i, value/len(preds)))
  return average

# Calculate metrics for all model predictions, returns a list
def evaluate_model(preds, target_paths_here, metric, color_threshold):
  result = []
  metric.reset_states()
  for i, mask in enumerate(preds):
      mask = np.argmax(mask, axis=2)
      mask = np.expand_dims(mask, axis=-1)
      target = cv2.imread(target_paths_here[i])
      target = cv2.cvtColor(target, cv2.COLOR_BGR2RGB)

      target = cv2.resize(target, img_size[::-1])
      target = extract_mask_from_ground(target, color_threshold)

      metric.update_state(target, mask)
      result.append(metric.result().numpy())
  return result

def eval_and_log(preds, img_paths, run_id, metric_name="", color_threshold = lower_color_threshold):
  # Define metrics
  meanIoU = tf.keras.metrics.MeanIoU(num_classes=4)
  accuracy = tf.keras.metrics.Accuracy()
  rmse = tf.keras.metrics.RootMeanSquaredError()

  # Calculate metrics for validation set
  dice = np.mean(calculate_dice_loss(preds, img_paths, color_threshold))
  iou = np.mean(evaluate_model(preds, img_paths, meanIoU, color_threshold))
  accuracy =  np.mean(evaluate_model(preds, img_paths, accuracy, color_threshold))
  rmse = np.mean(evaluate_model(preds, img_paths, rmse, color_threshold))

  metrics = [accuracy, dice, iou,  rmse]

  #Log parameters with mlflow
  if run_id != 0:
    client.log_metric(run_id, "{} accuracy".format(metric_name), accuracy)
    client.log_metric(run_id, "{} iou".format(metric_name), iou)
    client.log_metric(run_id, "{} dice".format(metric_name), dice)
    client.log_metric(run_id, "{} rmse".format(metric_name), rmse)

  # Display metrics
  print("Accuracy on {} set: {}".format(metric_name, accuracy))
  print("Mean IoU on {} set: {}".format(metric_name, iou))
  print("Dice score on {} set: {}".format(metric_name, dice))
  print("RMSE on {} set: {}".format(metric_name,rmse))
  print()

  return metrics

if __name__ == '__main__':

    # Set params for plots with pyploy
    plt.style.use('seaborn-pastel')
    params = {"ytick.color" : "black",
            "xtick.color" : "black",
            "axes.labelcolor" : "black",
            "axes.edgecolor" : "black",
            "text.usetex" : True,
            "font.family" : "serif",
            "font.serif" : ["Computer Modern Serif"]}
    plt.rcParams.update(params)

    # Prepare paths of input images and target segmentation masks
    input_dir = "./render/"
    target_dir = "./ground/"

    test_input_dir = "./real_moon_images/input/"
    test_target_dir = "./real_moon_images/target/."

    img_size = (320, 320)          #This is the biggest image I could get
    num_classes = 4
    batch_size = 16                # Bigger sizes sometimes couse crashes due to low memory
    lower_color_threshold = 100

    # Initialize in order to prevent error when loading model for inference only
    run_id = 0

    # Sorting image names in input folder
    input_img_paths = sorted([
        os.path.join(input_dir, fname)
        for fname in os.listdir(input_dir)
        if fname.endswith(".png")
    ])

    #Sorting mask names in target folder
    target_img_paths = sorted([
        os.path.join(target_dir, fname)
        for fname in os.listdir(target_dir)
        if fname.endswith(".png") and not fname.startswith(".")
    ])

    # Sorting images in real data input folder
    input_real_paths = sorted([
        os.path.join(test_input_dir, fname)
        for fname in os.listdir(test_input_dir)
        if fname.endswith(".png")
    ])

    # Sorting images in real data targte folder
    target_real_paths = sorted([
        os.path.join(test_target_dir, fname)
        for fname in os.listdir(test_target_dir)
        if fname.endswith(".png")
    ])

    print("Number of input samples:", len(input_img_paths))
    print("Number of target samples:", len(target_img_paths))

    #Input and target images are now corresponding
    for input_path, target_path in zip(input_img_paths[:10], target_img_paths[:10]):
        print(input_path, "|", target_path)


    print("Number of real input samples", len(input_real_paths))
    print("Nmber of real target samples", len(target_real_paths))
    for input_path, target_path in zip(input_real_paths[:10], target_real_paths[:10]):
        print(input_path, "|", target_path)

    clean_target_dir = "./clean/"
    ground_target_dir = "./ground/"

    # Sorting image names in clean folder
    clean_target_img_paths = sorted([
        os.path.join(clean_target_dir, fname)
        for fname in os.listdir(clean_target_dir)
        if fname.endswith(".png")
    ])

    # Sorting images in ground folder
    ground_target_img_paths = sorted([
        os.path.join(ground_target_dir, fname)
        for fname in os.listdir(ground_target_dir)
        if fname.endswith(".png")
    ])


    print("Number of clean target samples:", len(input_img_paths))
    print("Number of ground target samples:", len(target_img_paths))

    for input_path in clean_target_img_paths[:10]:
        print(input_path)

    for input_path in ground_target_img_paths[:10]:
        print(input_path)

    # Always take first 1000 images as a test set
    test_samples = 1000

    # Define test set input paths with corresponding targte
    test_input_img_paths = input_img_paths[:test_samples]
    test_target_img_paths = target_img_paths[:test_samples]

    # Define test sets input paths for clean and ground masks
    test_clean_target_img_paths = clean_target_img_paths [:test_samples]
    test_ground_target_img_paths = ground_target_img_paths[:test_samples]

    # Paths for both training and evaluation sets
    val_train_input_img_paths = input_img_paths[test_samples:]
    val_train_target_img_paths = target_img_paths[test_samples:]


    # Split image paths into training and validation set
    val_samples = 1000
    random.Random(1337).shuffle(val_train_input_img_paths)
    random.Random(1337).shuffle(val_train_target_img_paths)
    train_input_img_paths = val_train_input_img_paths[:-val_samples]
    train_target_img_paths = val_train_target_img_paths[:-val_samples]
    val_input_img_paths = val_train_input_img_paths[-val_samples:]
    val_target_img_paths = val_train_target_img_paths[-val_samples:]

    # Instantiane data Sequences for each split
    train_gen = LunarLandscapeAgumentation(
                batch_size, img_size, train_input_img_paths, train_target_img_paths, lower_color_threshold)

    val_gen = LunarLandscapeAgumentation(
                batch_size, img_size, val_input_img_paths, val_target_img_paths, lower_color_threshold)

    test_gen = LunarLandscape(
        batch_size, img_size, test_input_img_paths, test_target_img_paths, lower_color_threshold)

    clean_test_gen = LunarLandscape(
        batch_size, img_size, test_input_img_paths, test_clean_target_img_paths, color_threshold = 0)

    ground_test_gen = LunarLandscape(
        batch_size, img_size, test_input_img_paths, test_ground_target_img_paths, color_threshold = 0)
        

    # Free up RAM in case the model definition cells were run multiple times
    keras.backend.clear_session()

    #Build the model
    # model = get_model(img_size, num_classes)
    model = get_linknet_model(img_size, num_classes) # Ask how to properly change the models here
    model.summary()


    mlflow.autolog()

    # Use this one with Unet
    model.compile(
        'Adam',
        loss=sm.losses.dice_loss,
        metrics=[sm.metrics.iou_score],
    )
    # Define callbacks
    callbacks = [
        keras.callbacks.EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True),
        keras.callbacks.ModelCheckpoint("Linknet_mobilenetv2_320x320.h5", save_best_only=True),
    ]

    epochs = 80
    with mlflow.start_run(run_name="Linknet with mobilenetv2 backbone, threshold 100") as run:
        history = model.fit(train_gen, epochs=epochs, validation_data=val_gen, callbacks=callbacks)

    history_df = pd.DataFrame(history.history)
    hist_csv_file = 'history.csv'
    history_df = pd.read_csv('history.csv')

    fig, (ax1, ax2) = plt.subplots(2, figsize=(12,10))

    ax1.plot(history_df['loss'], label="loss")
    ax1.plot(history_df['val_loss'], label="val_loss")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Dice loss")
    ax1.legend()


    ax2.plot(history_df['iou_score'], label="IoU")
    ax2.plot(history_df['val_iou_score'], label="val IoU")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("IoU")
    ax2.legend()

    converter = tf.lite.TFLiteConverter.from_saved_model(temp_dir + '/data/model') # path to the SavedModel directory
    converter.optimizations = [tf.lite.Optimize.DEFAULT]

    # with mlflow.start_run(run_name="LinkNet with MobileNetV2, compressed") as run:
    tflite_model = converter.convert()
    interpreter = tf.lite.Interpreter(model_content=tflite_model)
    interpreter.allocate_tensors()
    test_preds_compressed = compressed_predict(interpreter, test_input_img_paths)
    real_preds_compressed = compressed_predict(interpreter, input_real_paths)
    val_preds_compressed = compressed_predict(interpreter, val_input_img_paths)
    clean_test_preds_compressed =  test_preds_compressed 
    ground_test_preds_compressed =  test_preds_compressed 
    val_gen = LunarLandscape(batch_size, img_size, val_input_img_paths, val_target_img_paths, lower_color_threshold)
    val_preds = model.predict(val_gen)

    #Generate predictions for real moon images
    real_gen = LunarLandscape(batch_size, img_size, input_real_paths, target_real_paths, 5)
    real_preds = model.predict(real_gen)

    #Generate predictions for test sets
    test_preds = model.predict(test_gen)

    # Generate predictions for test set with clean masks
    clean_test_preds = model.predict(clean_test_gen)

    # Generate predictions for test set with uncleaned masks
    ground_test_preds = model.predict(ground_test_gen)
    val_metrics = eval_and_log(val_preds, val_target_img_paths, run_id, "val")
    test_metrics = eval_and_log(test_preds, test_target_img_paths, run_id, "test")
    clean_test_metrics = eval_and_log(clean_test_preds, test_clean_target_img_paths, run_id, "clean test")
    ground_test_metrics = eval_and_log(ground_test_preds, test_ground_target_img_paths, run_id, "ground test")
    real_metrics = eval_and_log(real_preds, target_real_paths, run_id, "real")

    val_metrics_compressed = eval_and_log(val_preds_compressed, val_target_img_paths, run_id, "val")
    test_metrics_compressed = eval_and_log(test_preds_compressed, test_target_img_paths, run_id, "test")
    clean_test_metrics_compressed = eval_and_log(clean_test_preds_compressed, test_clean_target_img_paths, run_id, "clean test")
    ground_test_metrics_compressed = eval_and_log(ground_test_preds_compressed, test_ground_target_img_paths, run_id, "ground test")
    real_metrics_compressed = eval_and_log(real_preds_compressed, target_real_paths, run_id, "real")
