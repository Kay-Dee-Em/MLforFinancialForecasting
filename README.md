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

* `image_generator.py` - class for generating GAF images based on raw time series (e.g., `OIH_adjusted.txt`)
* `nn_model.py` - class for training, validating, and testing neural networks based on input data - GAF images
* `calculate_metrics.py` - functions for choosing the best validation models, evaluating them on test data, and calculate their metrics

Auxiliary scripts:

* `nn_model.py` -> `nn_model_auxiliary.py` - auxiliary class (Convolutional Block Attention Module, StopOnPoint for breaking the training process), and function for NN (ReshapeLayer before LSTM layer)
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

### Approach (default parameters values)

Raw time series data (without headers):

<table>
<thead>
  <tr>
    <th>Date</th>
    <th>Time</th>
    <th>Open</th>
    <th>High</th>
    <th>Low</th>
    <th>Close</th>
    <th>Volume</th>
  </tr>
</thead>
<tbody>
  <tr>
    <td>...</td>
    <td>...</td>
    <td>...</td>
    <td>...</td>
    <td>...</td>
    <td>...</td>
    <td>...</td>
  </tr>
  <tr>
    <td>04/12/2011</td>
    <td>14:01</td>
    <td>862.79</td>
    <td>863.86</td>
    <td>862.79</td>
    <td>863.8</td>
    <td>4050</td>
  </tr>
  <tr>
    <td>04/12/2011</td>
    <td>14:02</td>
    <td>863.86</td>
    <td>864.36</td>
    <td>863.8</td>
    <td>864.36</td>
    <td>1577</td>
  </tr>
  <tr>
    <td>04/12/2011</td>
    <td>14:03</td>
    <td>864.25</td>
    <td>864.58</td>
    <td>864.19</td>
    <td>864.25</td>
    <td>1649</td>
  </tr>
  <tr>
    <td>...</td>
    <td>...</td>
    <td>...</td>
    <td>...</td>
    <td>...</td>
    <td>...</td>
    <td>...</td>
  </tr>
</tbody>
</table>

is cut to variables: `Date`, `Time` and `Close`, then aggreagted (id est - mean) into 4 different intervals: `1 hour`, `2 hours`, `4 hours` and `1 day` and keeping `20 days` (step equals 1 day but considered period equals 20 days), the last 20 data points is choosen for each of the 4 intervals. Each of the intervals is scaled, transormed into polar coordinates and then processed into Gramian Angular Difference Field image. Finally the 4 images (each 10X10X3) are concatenated into 1 (40X40X3). Decision about optimal investemnt strategy is determined. If Close price in 21th day is higher than Close price in 20th interal day, then the strategy is `LONG -> 1`, else `SHORT -> 0`.

Generating images (neural network input data) in graphs:

<p float="left">
  <img alt='line' src="https://github.com/Kay-Dee-Em/MLforFinancialForecasting/blob/main/PLOTS/LONG_2021_12_31_line.png" width="200" height="200"/>
  <img alt='polar' src="https://github.com/Kay-Dee-Em/MLforFinancialForecasting/blob/main/PLOTS/LONG_2021_12_31_polar.png" width="200" height="200"/> 
  <img alt='GADF' src="https://github.com/Kay-Dee-Em/MLforFinancialForecasting/blob/main/PLOTS/LONG_2021_12_31_GADF.png" width="200" height="200"/>
</p>

Generated images are used as input data to neural network model which default architecture is CNN+LSTM+Attention neural network...

<!---
Validation process chooses the best model accoring to each validation cateogry.
And finally the best models are tested.
-->
