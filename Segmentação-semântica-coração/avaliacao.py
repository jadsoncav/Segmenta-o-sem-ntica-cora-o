import os
import numpy as np
import cv2
import pydicom as dicom
from glob import glob
from tqdm import tqdm
import tensorflow as tf
from sklearn.metrics import accuracy_score, f1_score, jaccard_score, precision_score, recall_score
from indicadores import iou, coeficiente_dice, perda_dice
import keras

def criar_diretorio(caminho):
    if not os.path.exists(caminho):
        os.makedirs(caminho)

def processar_imagem(caminho_imagem, modelo):
    imagem = dicom.dcmread(caminho_imagem).pixel_array
    imagem = np.expand_dims(imagem, axis=-1)
    imagem = imagem / np.max(imagem) * 255.0
    x = imagem / 255.0
    x = np.concatenate([x, x, x], axis=-1)
    x = np.expand_dims(x, axis=0)

    mascara = modelo.predict(x)[0]
    mascara = (mascara > 0.5).astype(np.int32) * 255

    return imagem, mascara

if __name__ == "__main__":
    np.random.seed(42)
    tf.random.set_seed(42)

    criar_diretorio("DIRETÓRIO_PARA_TESTAR")

    modelo = tf.keras.models.load_model("CAMINHO_DO_MODELO", custom_objects={
        'iou': iou,
        'coeficiente_dice': coeficiente_dice,
        'perda_dice': perda_dice,
    })

    testar_x = glob("CAMINHO_DOS_DADOS")
    print(f"testar: {len(testar_x)}")

    for caminho_imagem in tqdm(testar_x):
        nome_dir = caminho_imagem.split("/")[-3]
        nome = nome_dir + "_" + caminho_imagem.split("/")[-1].split(".")[0]

        imagem, mascara = processar_imagem(caminho_imagem, modelo)
        imagens_concatenadas = np.concatenate([imagem, mascara], axis=1)
        cv2.imwrite(f"DIRETÓRIO_PARA_TESTAR/{nome}.png", imagens_concatenadas)
