## MLforFinancialForecasting (CREATION OF THE PACKAGE IN PROGRESS)

### Python 3.10 

**MLforFinancialForecasting** will be a mini Python package for financial time series classification. Optimal investment strategy aka prediction/classification is determined by the best set of neural networks (default architecture: CNN+LSTM + Attention mechanism neural network) and the sole input to a project is time series (default: `OIH_adjusted.txt`) which in result of processing is used for creation input Gramian Angular Field images.  

### Install packages

```
$ pip install -r requirements.txt
```

### Run all the procedure steps

```
$ python app.py
```

### Scripts


Main scripts:

* `ImageGenerator.py` - class for generating GAF images based on raw time series (e.g., `OIH_adjusted.txt`)
* `NNModel.py` - class for training, validating, and testing neural networks based on input data - GAF images
* `calculate_metrics.py` - functions for choosing the best validation models, evaluating them on test data, and calculate their metrics


Auxiliary scripts:

* `NNModel.py` -> `NNmodelAuxiliary.py` - auxiliary class (Convolutional Block Attention Module, StopOnPoint for breaking the training process), and function for NN (ReshapeLayer before LSTM layer)
* `calculate_metrics.py` -> `join_predictions.py` - join predictions when NN was passed externally or join all intervals' predictions
* `calculate_metrics.py` -> `func_validation.py` - functions for choosing the best validation models
* `calculate_metrics.py` -> `func_test.py` - function for evaluating the best validation models on test datasets, and calculating their metrics
* `create_plots.py` -> create a variety of plots


Results:

- `GAF/` -> GAF images for the time series dataset
- `PREDICTIONS/` -> neural networks' predictions
- `PREDICTIONS_MAX_PREDICTIONS/` -> all and max possible predictions outcomes for validation, and test intervals for each combination (default: 1023 (the 10th element vector) - all possible combinations for n-th element vector of initial neural networks' weights)
- `PREDICTIONS_DAILY_BEST_COMBINATIONS/` -> raw (binary) predictions for chosen validation models for test datasets for each method of evaluation
- `HEATMAPS/` -> heatmaps of possible results for chosen validation models for test datasets for the time series dataset
- `PLOTS/` -> examples of plots
