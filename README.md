# A Deep Inverse-Mapping Model For A Flapping Robotic Wing
Official PyTorch Implementation for the "A Deep Inverse-Mapping Model for a Flapping Robotic Wing" Paper (ICLR 2025)

<p align="center">
    <a href="https://arxiv.org/abs/1234.56789">
        <img src="https://img.shields.io/badge/Paper-arXiv-red" alt="Paper">
    </a>
    <a href="https://paperswithcode.com/paper/a-deep-inverse-mapping-model-for-a-flapping">
        <img src="https://img.shields.io/badge/Papers%20with%20Code-1234-green" alt="Papers with Code">
    </a>
</p>

## Setup

1. Clone the repository:
    ```bash
    git clone https://github.com/hadar933/AdaptiveSpectrumLayer.git
    cd repo
    ```

2. Create a virtual environment and activate it:
    ```bash
    python3 -m venv venv
    source venv/bin/activate
    ```
    <details>
    <summary>Optional: Configure the Python interpreter in VS Code</summary>

    Configure the Python interpreter in VS Code:
    - Press `Ctrl+Shift+P` to open the command palette.
    - Type `Python: Create Environment` and select `venv`

    </details>

3. Install the dependencies:
    ```bash
    pip install -r requirements.txt
    ```

## Structure

The project is organized into several directories, each with a specific purpose:
- **ML**  
  Implements core machine learning functionality using PyTorch. This includes:  
  - Preprocessed data from our flapping wing system.  
  - A **Zoo** subdirectory with neural network implementations, including the Adaptive Spectrum Layer (ASL) and other architectures.
  - Dataset handling.  
  - Normalization and training utilities (e.g., fit, predict, evaluate).  

- **Camera**  
  Handles raw image data and camera calibration. It includes scripts for:  
  - Extracting 3D coordinates.  
  - Tracking trajectories.  
  - MATLAB stereo camera setup calibrations

- **DataHandler**  
  Contains utilities for:  
  - Data encoding: e.g sin-cos encoding, or calculation of torques from forces.
  - Preprocessing: a basic class that performs resampling, interpolation, etc.
  - Synchronization scrips for matching signals by their start times and consequtive difference value.

- **Forces**  
  Processes raw data from force sensors and prepares it for downstream analysis.

- **LTSFLinear**  
  A fork of the primary paper implementation, this directory contains various neural network models, including NLinear, NLinear and FEDformer, for benchmarking.  


- **Utilities**  
  Contains miscellaneous scripts for:  
  - Plotting.  
  - Data splitting.  
  - Adding explainability to the analysis.

> **Note**  
>
>The `Camera` and `Forces` directories are primarily included for clarity >and completeness. These directories were used to generate the already >ready-to-use data located in the `ML/Data` directory. Users can directly >utilize the data in `ML/Data` without needing to refer to the aformentioned dirs.

> 

## Usage

To run a training job:
```bash
python ML/main.py
```

## Citation

If you find this code useful, consider citing our paper with:
```bibtex
@inproceedings{author2025deep,
  title={A Deep Inverse-Mapping Model for a Flapping Robotic Wing},
  author={Hadar Sharvit, Raz Karl, Tsevi Beatus},
  booktitle={International Conference on Learning Representations},
  year={2025}
}
```

## Acknowledgments

We would like to thank our collaborators and funding agencies for their support.