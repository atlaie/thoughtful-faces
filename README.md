# Thoughtful faces: the repository.

This repository contains the data preprocessing, computation, and plotting scripts used in the research paper: **"Thoughtful faces: inferring internal states across species using facial features"**. The purpose of these scripts is to provide a transparent and reproducible framework for our data analysis process, enabling peers to review, replicate, and build upon our work.

## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes.

### Prerequisites

Before running the scripts, ensure you have the required software/tools installed. You can check them out in "requirements_final.txt".

You can install all required Python packages using the following command:

```bash
pip install -r requirements_final.txt
```

### Installation

Clone this repository to your local machine using:

```bash
git clone https://github.com//atlaie/thoughtful-faces.git
```

Navigate into the cloned repository:

```bash
cd thoughtful-faces
```

### Running the Scripts

In the following, for every explanation involving a particular species, there's another mirrored script with the other one. For simplicity, we'll explain everything using "mouse" as an example, but it would be exactly the same if "macaque" was used instead.

1. **Data Preprocessing**: To prepare the data for analysis, run the preprocessing script:

```bash
python preprocess_mouse_data.py
```

We have made available online the output of these files; we are sharing the pre-processing scripts for transparency purposes.

2. **Computation**: There are two main computations and modes of use of these scripts: cross-validation mode (i.e. finding the parameters that work best to explain the data) and inference (after the previously mentioned parameters are found).
  
To use the cross-validation mode, use:

```bash
python -m mouse_MSLR_final --doCV "True" --date "DATE1"
```

Where "DATE1" is the current date (to append it to the output name).

To use the previously found parameters in cross-validation and then infer for the held out sets, use:

```bash
python -m mouse_MSLR_final --doCV "False" --filename "path/to/file/Mouse/Results_CV_MSLR_Optuna_mouse_NTRIALStrials_Ns_states_DATE1_RT_AllSubjects_R2score_CVOnly.npz" --date "DATE2"
```

With "Ns" being the optimal number of states (found in cross-validation), "NTRIALS" is the number of trials you used in the optimization step (100 by default); "DATE1" being the CV date and "DATE2" the inferring date (not necessarily matching). Results will be automatically saved in a new folder called "Results", within the same parent directory.

3. **Plotting**: For generating the figures presented in the paper, use:

```bash
python plotting_main.py
```

And for the supplementary figures:

```bash
python plotting_supplementary.py
```

## Structure of the Repository

- `preprocess_mouse_data.py`: Script for cleaning and preparing mouse data.
- `mouse_MSLR_final.py`: Script containing the main computational analyses for mouse data.
- `plotting_main.ipynb`: Notebook for creating visualizations.
- `plotting_supplementary.ipynb`: Notebook for creating supplementary visualizations.
- `/data`: Directory containing raw and processed data files.

## Reproducing Results

To reproduce the results presented in the paper, follow the execution order mentioned in the [Running the Scripts](#running-the-scripts) section. Ensure the `/data` directory contains the necessary datasets.

## Authors

- **Alejandro Tlaie** - *Initial work* - [atlaie](https://github.com/atlaie)

See also the list of [contributors](https://github.com/yourusername/your-repository-name/contributors) who participated in this project.

## Citation

If you find this work useful, please cite our paper:

```bibtex
@article{tlaie2024thoughtful,
  title={Thoughtful faces: inferring internal states across species using facial features},
  author={Tlaie, Alejandro and Abd El Hay, Muad Y and Mert, Berkutay and Taylor, Robert and Ferracci, Pierre-Antoine and Shapcott, Katharine and Glukhova, Mina and Pillow, Jonathan W and Havenith, Martha N and Sch{\"o}lvinck, Marieke},
  journal={bioRxiv},
  pages={2024--01},
  year={2024},
  publisher={Cold Spring Harbor Laboratory}
}
```
