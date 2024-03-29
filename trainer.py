from model import *
from data import *
import tensorflow as tf
import mlflow
import argparse
from predictor import Infer
from metrics import flow_mean_iou

mlflow.tensorflow.autolog()

# Parse command-line arguments
parser = argparse.ArgumentParser(description="Train a UNet model on membrane data.")
parser.add_argument(
    "--rotation_range",
    type=float,
    default=0.2,
    help="Rotation range for data augmentation",
)
parser.add_argument(
    "--width_shift_range",
    type=float,
    default=0.05,
    help="Width shift range for data augmentation",
)
parser.add_argument(
    "--height_shift_range",
    type=float,
    default=0.05,
    help="Height shift range for data augmentation",
)
parser.add_argument(
    "--shear_range", type=float, default=0.05, help="Shear range for data augmentation"
)
parser.add_argument(
    "--zoom_range", type=float, default=0.05, help="Zoom range for data augmentation"
)
parser.add_argument(
    "--horizontal_flip",
    type=bool,
    default=True,
    help="Whether to apply horizontal flip for data augmentation",
)
parser.add_argument(
    "--fill_mode", type=str, default="nearest", help="Fill mode for data augmentation"
)
parser.add_argument(
    "--steps_per_epoch", type=int, default=2, help="Shear range for data augmentation"
)
parser.add_argument(
    "--epochs", type=int, default=1, help="Shear range for data augmentation"
)

args = parser.parse_args()

# Data generator arguments
data_gen_args = dict(
    rotation_range=args.rotation_range,
    width_shift_range=args.width_shift_range,
    height_shift_range=args.height_shift_range,
    shear_range=args.shear_range,
    zoom_range=args.zoom_range,
    horizontal_flip=args.horizontal_flip,
    fill_mode=args.fill_mode,
)
myGene = trainGenerator(
    2, "data/membrane/train", "image", "label", data_gen_args, save_to_dir=None
)

model = unet()
model_checkpoint = tf.keras.callbacks.ModelCheckpoint(
    "unet_membrane.keras", monitor="loss", verbose=1, save_best_only=True
)
model.fit(
    myGene,
    steps_per_epoch=args.steps_per_epoch,
    epochs=args.epochs,
    callbacks=[model_checkpoint],
)

# create pyfunc model def for logging
# artifacts = {"Binarizer": "unet_membrane.keras"}
# class UnetMlfowDef(mlflow.pyfunc.PythonModel):
#     def load_context(self, context):
#         from predictor import Infer
#         flow_model_path = context.artifact['Binarizer']
#         self.engine = Infer(flow_model_path)

#     def predict(self, context, model_input, params = {'thresh': 0.2}):
#         thresh=  params['thresh']
#         pred = self.engine.predictor(model_input, thresh)
#         #  TODO convert pred to numpy
#         return pred
# # get model sig
# from mlflow.models import infer_signature
# cl = Infer('unet_membrane.keras')
# testGene = myGene
# inst, lbl = next(iter(testGene))
# pred = cl.predictor(inst, 0.2)
# sig = infer_signature(inst, pred)

# # convert generator to numpy array
# num_samples = 10
# batch_size = 2
# samples = []
# labels = []
# for _ in range(num_samples):
#     batch_samples, batch_labels = next(testGene)
#     samples.append(batch_samples)
#     labels.append(batch_labels)
# samples_arr = np.concatenate(samples, axis=0)
# labels_arr = np.concatenate(labels, axis=0)
# # create mlflow dataset instances
# flow_ds = mlflow.data.from_numpy(features = samples_arr, targets = labels_arr, name ='sample_ds')

# # eval fun
# from mlflow.models import MetricThreshold
# thresholds = {
#     "flow_iou_metric": MetricThreshold(
#         # accuracy should be >=0.8
#         threshold=0.2,
#         # accuracy should be at least 5 percent greater than baseline model accuracy
#         # min_absolute_change=0.05,
#         # accuracy should be at least 0.05 greater than baseline model accuracy
#         # min_relative_change=0.05,
#         greater_is_better=True,
#     ),
# }
# flow_iou = mlflow.metrics.make_metric(eval_fn = flow_mean_iou, greater_is_better = True, name=  "flow_iou_metric")

# # log extra params with autologger
# mlflow.autolog(disable=True)
# act_run = mlflow.last_active_run()
# act_run_id = act_run.info.run_id
# code_dep_path = 'C:\\Users\\Administrator\\mlflow_projects\\Unet_mlflow\\predictor.py'
# with mlflow.start_run(run_id=act_run_id):
#     # first eval the model
#     mlflow.evaluate(model=UnetMlfowDef, data= samples, targets = labels, extra_metrics=[flow_iou], validation_thresholds=thresholds)

# mlflow.pyfunc.log_model(artifact_path = 'pyfunc_artifacts', code_path = [code_dep_path],
#                         python_model = UnetMlfowDef(), artifacts = artifacts,
#                         signature = sig, metadata=  "default training provided by repo")


# testGene = testGenerator("data/membrane/test")
# results = model.predict_on_batch(testGene)
# saveResult("data/membrane/test",results)
