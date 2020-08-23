import os
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

str_yen = "//"

def get_current_dir():
  tmp_folder_path = os.path.dirname(__file__)
  tmp_folder_path = tmp_folder_path.replace(str_yen, "/")
  return tmp_folder_path

def show_figure(file_path_relative, figsize=(5, 5), dpi=200):
  foldr_path_top = get_current_dir()
  file_path_relative = file_path_relative.replace("./", "")
  file_path_abs = foldr_path_top + "/" + file_path_relative
  im = Image.open(file_path_abs)
  im_np = np.array(im)
  plt.figure(figsize=figsize, dpi=dpi)
  plt.imshow(im_np)
  plt.show()

def show_figure2(file_path_relative):
  foldr_path_top = get_current_dir()
  file_path_relative = file_path_relative.replace("./", "")
  file_path_abs = foldr_path_top + "/" + file_path_relative
  im = Image.open(file_path_abs)
  im_np = np.array(im)
  plt.figure(figsize=(5, 5), dpi=200)
  plt.imshow(im_np)
  plt.show()