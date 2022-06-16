### MLforFinancialForecasting

**Python 3.10**


Run all the procedure steps:

*app.py*


Main scripts:

* *ImageGenerator.py* - class for generating GAF images based on raw time series (e.g., *OIH_adjusted.txt*)
* *NNModel.py* - class for training, validating, and testing neural networks based on input data - GAF images
* *calculate_metrics.py* - functions for choosing the best validation models, evaluating them on test data, and calculate their metrics


Auxiliary scripts:

* *NNModel.py* -> *NNmodelAuxiliary.py* - auxiliary class (Convolutional Block Attention Module, StopOnPoint for breaking the training process), and function for NN (ReshapeLayer before LSTM layer)
* *calculate_metrics.py* -> *join_predictions.py* - join predictions when NN was passed externally or join all intervals' predictions
* *calculate_metrics.py* -> *func_validation.py* - functions for choosing the best validation models
* *calculate_metrics.py* -> *func_test.py* - function for evaluating the best validation models on test datasets, and calculating their metrics
* *create_plots.py -> create a variety of plots*


Results:

- *GAF/* -> GAF images for the OIH dataset
- *HEATMAPS/* -> heatmaps of possible results for chosen validation models for test datasets for the OIH dataset
- *PLOTS/* -> examples of plots
- *PREDICTIONS/* -> neural networks' predictions
- *PREDICTIONS_DAILY_BEST_COMBINATIONS/* -> raw (binary) predictions for chosen validation models for test datasets for each method of evaluation
- *PREDICTIONS_MAX_PREDICTIONS/* -> all and max possible predictions outcomes for validation, and test intervals for each of 1023 (all possible combinations for 10th element vector of initial neural networks' weights, the 10th element vector is changeable)
