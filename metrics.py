import mlflow
import numpy as np
import pandas as pd


"""
ONE PROBLEM WITH PYFUNC IS IT ONLY ACCEPTS 2d INPUTS
INCASE OF MULTIDIM YOU CAN USE DIFFERENT FLAVORS EG TF BUT
THEN YOU CANNOT ADD THE PIPELINE ONLY MODEL INPUTS AND OUTPUTS CAN 
BE EVAL W/O ANY POSTPROCESS
WAYAROUND: save the preporcess data seperately then use the model passing
afterwards, in metrics post process the data and transform it before evaluating
or 
only use model in and out eval
"""


def iou(mask1, mask2):
    intersection = np.logical_and(mask1, mask2)
    union = np.logical_or(mask1, mask2)
    iou_score = np.sum(intersection) / np.sum(union)
    return iou_score


# to pass extra params use eval config->  col mapping
# pred and targets are pandas series
# can aslo take another metrics as param if you suply the metric name as parameter
def flow_mean_iou(predictions, targets):
    iou_scores = []

    # TODO: if the pyfunc only supports 2d inputs and you are now refeerring to tf
    # you will now need to do postprocessing

    for index, (pred_mask, target_mask) in enumerate(zip(predictions, targets)):
        iou_score = iou(pred_mask, target_mask)
        iou_scores.append(iou_score)

    iou_series = pd.Series(iou_scores)

    return mlflow.metrics.MetricValue(
        scores=iou_series, aggregate_results={"mean_iou": iou_series.mean()}
    )
