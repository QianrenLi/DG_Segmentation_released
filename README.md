# Project 1 for advanced signal processing

## Usage
+ data file: put file in `./data/`, write your detail link --- training set, validation set, testing set --- into `input_pattern`, `val_input_pattern` and `test_input_pattern`. or by modifying input of `load_dataset`
+ Training and result display:
  + Run `project1.ipynb`
  + If want check loss curve, run `loss_display.py`

## Dependencies
+ `pip install -r requirements.txt`
## File Description
+ results: 
  + `./results/Loss` contains loss information for each epoch in `txt` format which can be read by `loss_display`
  + `./results/model` contains the final model after domain generalization and data augmentation.
  + `./results/DG_result` contains the DG result when the phase spectrum is used.
- `project1.ipynb` contains results of domain generalization, the feature space representation, training, validation and testing diagram.
- `dataset.py` this file mainly contains `load_dataset` function to load images with typical dataset format.  Function domain generalization and data augmentation based on transformer is implemented within this file which could be applied by changing input of `load_dataset`.
- `dis_rep.py` contains distance metric like 2-norm, 1-norm, CS-distance SNR, P-SNR, SSIM, intra-clustering distance, inter-clustering distance.
- `Result_disp.py` the training result is saved in this file and then be plotted in histogram.
- `loss_display.py` used to plot loss curve.
- `pro1.py` is an old training python (not valid now).
- `K_fold_validation.py` is the K fold validation used to save weights and train the model.
- `DG distance.py`: file used to generate the distance between generalized image from 3 domain and the source image (not valid now)
- `test.py`: only for testing (not valid now)
