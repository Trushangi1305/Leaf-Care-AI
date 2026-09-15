# AgriSmart AI - Trained Crop Disease Baseline

This model was trained on the supplied dataset's `part_1/color` images only. It covers 38 classes.

## Evaluation
- Samples used: 9,154
- Train split: 7,323
- Test split: 1,831
- Accuracy: 63.63%
- Features: HOG + HSV color histogram
- Classifier: SGD logistic-loss classifier

This is a lightweight CPU baseline, not a deep-learning model. The accuracy is reported honestly on a held-out stratified test split. For a production-quality model, use transfer learning (e.g. EfficientNet/MobileNet) on a GPU and test on real smartphone/farm images.

## Prediction
From this directory:
`python prediction_model/predict.py path/to/leaf.jpg`

## Retraining
Place the extracted dataset at `dataset/part_1/color` and run:
`python train_model.py`
