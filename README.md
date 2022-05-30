# U-Net\#

Custom U-Net type architecture for multi-class nuclei segmentation.

## Data

For this project we used PanNuke dataset

To download the dataset and prepare folder structure run:
```sh
bash prepare_dataset.sh
```

## Usage

Model checkpoints are saved in 'saved/lightning_logs/version_XX/checkpoints'. 
Provide a full path to model checkpoint when running on a test set.

Predictions on the test set could be found in 'saved/preds/preds.npy'.

### Training
```sh
python3 src/cli.py fit
```
### Testing
```sh
python3 src/cli.py test --ckpt_path PATH_TO_SELECTED_CKPT_FILE
```
### Calculate test metrics
```sh
python3 src/metrics.py PATH_TO_PREDS_NPY
```
Calculated metrics are saved in folder 'saved'.

