# PCE
This repository contains the data and code associated with the paper: https://arxiv.org/abs/2401.09421.

### Dependencies:

The project relies on the following libraries: `numpy`, `autograd`, `qibo`, `qiskit`, `torch`, `tensorflow`, `tensorly-quantum`, `MQLib` (installation might be challenging), `networkx`, `sqlite`, `gradient_free_optimizers`, `scipy`, and `hypermapper`.

### Usage:

A working example of the code can be found in the `example.ipynb` file, which includes a brief explanations for the various flags used. Note that if you are using a gradient-based optimizer, you might need to modify the `numpy` import to `autograd.numpy` within the base code of `qibo`.

### Experimental Data:

The `experiment` folder contains all experimental data along with the corresponding data analysis.

### Notes:

The code is convoluted and not optimized. If you encounter any issues or have questions, please don't hesitate to reach out for assistance
