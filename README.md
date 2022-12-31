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

## Training Result

> + See more: [project1.ipynb](./project1.ipynb);
> + You can download our pre-trained model at <https://github.com/QianrenLi/ad_sig_pro1/releases/tag/v1>.

|                   | **Verification DICE** | **Test1 DICE** | **Test2 DICE** | **Test3 DICE** | **Test1 HD95** | **Test2 HD95** | **Test3 HD95** |
| ----------------- | --------------------- | -------------- | -------------- | -------------- | -------------- | -------------- | -------------- |
| **Cross Entropy** | 0.172                 | 0.080          | 0.183          | 0.147          | 22.198         | 4.070          | 7.992          |
| **DICE1**         | 0.404                 | 0.115          | 0.363          | 0.289          | 11.689         | 2.596          | 4.136          |
| **DICE2**         | 0.873                 | 0.654          | 0.869          | 0.780          | 3.846          | 2.402          | 4.131          |
| **DICE+CE**       | 0.893                 | 0.654          | 0.899          | 0.805          | 13.055         | 1.948          | 4.140          |
| **DA+DICE+CE**    | 0.910                 | 0.733          | 0.913          | 0.865          | 4.166          | 1.534          | 2.486          |
| **DG+DICE+CE**    | 0.901                 | 0.715          | 0.907          | 0.849          | 3.707          | 1.618          | 2.821          |
| **DG+DA+DICE+CE** | 0.907                 | 0.737          | 0.909          | 0.889          | 3.723          | 1.626          | 1.892          |
