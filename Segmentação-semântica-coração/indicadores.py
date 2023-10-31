import numpy as np
import tensorflow as tf
import keras

def iou(y_verdadeiro, y_predito):
    """
    Calcula o Índice de Jaccard (IoU) entre as máscaras de verdadeiro e predito.

    Args:
        y_verdadeiro: Máscara de verdadeiro (ground truth).
        y_predito: Máscara predita.

    Returns:
        O valor do IoU.
    """
    intersecao = tf.reduce_sum(tf.cast(y_verdadeiro * y_predito, tf.float32))
    uniao = tf.reduce_sum(tf.cast(y_verdadeiro + y_predito, tf.float32)) - intersecao
    iou = (intersecao + 1e-15) / (uniao + 1e-15)
    return iou

def coeficiente_dice(y_verdadeiro, y_predito):
    """
    Calcula o Coeficiente de Dice entre as máscaras de verdadeiro e predito.

    Args:
        y_verdadeiro: Máscara de verdadeiro (ground truth).
        y_predito: Máscara predita.

    Returns:
        O valor do Coeficiente de Dice.
    """
    y_verdadeiro = tf.keras.layers.Flatten()(y_verdadeiro)
    y_predito = tf.keras.layers.Flatten()(y_predito)
    intersecao = tf.reduce_sum(y_verdadeiro * y_predito)
    dice = (2.0 * intersecao + 1e-15) / (tf.reduce_sum(y_verdadeiro) + tf.reduce_sum(y_predito) + 1e-15)
    return dice

def perda_dice(y_verdadeiro, y_predito):
    """
    Calcula a perda (loss) com base no Coeficiente de Dice.

    Args:
        y_verdadeiro: Máscara de verdadeiro (ground truth).
        y_predito: Máscara predita.

    Returns:
        O valor da perda com base no Coeficiente de Dice.
    """
    return 1.0 - coeficiente_dice(y_verdadeiro, y_predito)
