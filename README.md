# Thoughtful Faces: The Repository

This repository contains the data preprocessing, computation, and plotting scripts used in the research paper: **"Thoughtful Faces: Inferring Internal States Across Species Using Facial Features"**. These scripts aim to provide a transparent and reproducible framework for our data analysis process, enabling peers to review, replicate, and build upon our work.

## Getting Started

Follow these instructions to set up and run the project on your local machine for development and testing purposes.

### Prerequisites

Ensure you have Python installed (preferably version 3.9.6) and a suitable package manager like `conda`.

### Installation

1. Clone this repository to your local machine:

   ```bash
   git clone https://github.com/atlaie/thoughtful-faces.git
   ```

2. Navigate into the cloned repository:

   ```bash
   cd thoughtful-faces
   ```

3. Install the required Python packages listed in `requirements_final.txt`:

   ```bash
   conda create --name <env> --file <this file>
   ```

### Running the Scripts

For simplicity, this guide uses "mouse" as an example. Equivalent scripts exist for "macaque," and the instructions are identical, substituting "mouse" with "macaque" as needed.

#### 1. **Data Preprocessing**

To prepare the data for analysis, run the preprocessing script:

```bash
python preprocess_mouse_data.py
```

The outputs of these scripts are shared online for convenience, but the preprocessing scripts are provided for transparency.

#### 2. **Computation**

The computational scripts have two main modes:

- **Cross-validation mode**: Identifies optimal parameters for explaining the data.
- **Inference mode**: Applies these parameters to make predictions on held-out datasets.

To run cross-validation, use:

```bash
python -m mouse_MSLR_final --doCV "True" --date "DATE1"
```

Replace `DATE1` with the current date to append it to the output name.

To use the parameters identified during cross-validation for inference, use:

```bash
python -m mouse_MSLR_final --doCV "False" --filename "path/to/file/Mouse/Results_CV_MSLR_Optuna_mouse_NTRIALStrials_Ns_states_DATE1_RT_AllSubjects_R2score_CVOnly.npz" --date "DATE2"
```

Replace:
- `Ns` with the optimal number of states from cross-validation.
- `NTRIALS` with the number of trials (default: 100).
- `DATE1` with the cross-validation date.
- `DATE2` with the inference date (not necessarily matching).

Results will be saved in a `Results` folder within the repository.

#### 3. **Plotting**

To recreate the paper's figures, run:

```bash
python plotting_main.py
```

To generate supplementary figures, use:

```bash
python plotting_supplementary.py
```

## Repository Structure

- `preprocess_mouse_data.py`: Script for cleaning and preparing mouse data.
- `mouse_MSLR_final.py`: Script containing the main computational analyses for mouse data.
- `plotting_main.ipynb`: Notebook for creating main visualizations.
- `plotting_supplementary.ipynb`: Notebook for creating supplementary visualizations.
- `control_analyses.ipynb`: Notebook for creating and plotting control analyses.
- `/data`: Directory containing raw and processed data files.

## Reproducing Results

To reproduce the results from the paper:
1. Preprocess the data using the relevant script.
2. Perform cross-validation and save the optimal parameters.
3. Use the identified parameters to run inference.
4. Generate figures using the plotting scripts.

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
