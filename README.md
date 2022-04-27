# IFT6759 - Distribution shifts in wilds - iwildcam
 Our IFT6759 course work project is based on the paper [WILDS: A Benchmark of in-the-Wild Distribution Shifts](https://arxiv.org/abs/2012.07421). We have reproduced results based on existing code and ran experiments with data augment techniques such as Strong, Weak, Cutout and Cutmix techniques along with using Resnet and Densenet model architectures.
 * The code in this repo is based on the following repo: [wilds](https://github.com/p-lambda/wilds)
 * To replicate the results, we suggest running the code on COLAB in the same way as in **Run_Experiments.ipynb**.
 * Phase 1 and Phase 2 results are stored in [Results](https://drive.google.com/drive/folders/1fm8zeCJ55rNdDw4QKB5tXgExpsFx1ddK?usp=sharing) and Refer to plan.xlsx for the exact commands used for running experiments.
 * Refer to **Data_augument.ipynb** to see how the augment techniques works on a sample image.
 * Refer to Results_interpretation folder to see some Interesting observations during our learning phase.
 * For general functioning of the code refer to the aforementioned original github repo.
 
 The main differences between our version and the original:
  - Support for ERM, DeepCORAL and IRM algorithms exclusively.
  - Support for iwildcam dataset exclusively
  - Added support for Cutout, Cutmix data augmentations.
  - Extended weak data augmentation into weak enhanced, which adds horizontal and vertical translations.

  ## Installation

  ```bash
  git clone git@github.com:vijayakuruba/IFT6759_Distribution_shifts_WILDS.git
  cd wilds
  pip install -e .
  ```

  ### Requirements
  The WILDS package depends on the following requirements:

  - numpy>=1.19.1
  - ogb>=1.2.6
  - outdated>=0.2.0
  - pandas>=1.1.0
  - pillow>=7.2.0
  - pytz>=2020.4
  - torch>=1.7.0
  - torchvision>=0.8.2
  - tqdm>=4.53.0
  - scikit-learn>=0.20.0
  - scipy>=1.5.4

  Running `pip install wilds` or `pip install -e .` will automatically check for and install all of these requirements.

  Our code supports the optional use of [Weights & Biases](https://wandb.ai/site) to track and monitor experiments.
  To install the Weights and Biases Python package, run `pip install wandb`.


  ## Using the example scripts
In `examples/`, we provide a set of scripts that can be used to train models on the WILDS datasets.

```bash
#without augmentation
python examples/run_expt.py --dataset iwildcam --algorithm ERM --root_dir data

#with cutout data augmentation
python examples/run_expt.py --dataset iwildcam --algorithm deepCORAL --additional_train_transform cutout --root_dir data

#with weak+ data augmentation
python examples/run_expt.py --dataset iwildcam --algorithm deepCORAL --additional_train_transform weak --root_dir data

#with cutmix data augmentation(with beta function parameters at 0.4)
python examples/run_expt.py --dataset iwildcam --algorithm ERM --mixcut 0.4 --root_dir data
```

### Downloading and training on the WILDS dataset
The first time you run these scripts, you might need to download the datasets. You can do so with the `--download` argument, for example:
```python
# downloads (labeled) dataset
python examples/run_expt.py --dataset iwildcam --algorithm deepCORAL --root_dir data --download

```

Alternatively, you can use the standalone `wilds/download_datasets.py` script to download the datasets, for example:

```bash
# downloads (labeled) data
python wilds/download_datasets.py --root_dir data
```

This will download all datasets to the specified `data` folder.


### Working on subset of dataset

Since running fulldatset experiment is computationally expensive, we can check model performance with a subset of the data. 

```python
# run with subset of 40% dataset 
python run_expt.py --root_dir data --dataset iwildcam --algorithm ERM --seed 0 --lr 3e-05 --weight_decay 0 --frac 0.4 --model densenet121 --target_resolution 448 448 --n_epochs 3 --additional_train_transform weak 


```
