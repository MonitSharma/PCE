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



### How to Install MQLib

MQLib is a library for solving the Maximum Cut (Max-Cut) problem. Follow these steps to install and build it.

---

### Prerequisites

Before building MQLib, ensure you have the following tools installed on your system:

1. GNU Compiler Tools:
   - g++ (GNU C++ Compiler)
   - ar (GNU Archive Utility)

2. Common UNIX Utilities:
   - make (Build Automation Tool)
   - find, sed, and rm (Basic Utilities)

3. Git:
   - Required to clone the repository.

---

### Installation Steps

#### Step 1: Clone the Repository

1. Open a terminal (Linux/macOS) or a compatible environment (e.g., Cygwin, MSYS2, or WSL for Windows).
2. Run the following command to clone the MQLib repository:
```
   git clone https://github.com/MQLib/MQLib
```
3. Navigate to the cloned directory:
```
   cd MQLib
```

---

#### Step 2: Install Dependencies

##### Linux/macOS:
1. Install the required tools using your package manager:
```
   sudo apt update (For Ubuntu/Debian)
   sudo apt install build-essential git
```
   Or for macOS (using Homebrew):
```   
   brew install gcc make git
```

##### Windows:
Install a Unix-like environment such as MSYS2 or Cygwin:
- For MSYS2:
```
  pacman -Syu
  pacman -S base-devel git
```

- For Cygwin:
```
  Ensure gcc-g++, make, git, findutils, and sed are installed during setup.
```
---

#### Step 3: Build the Project

1. Run the make command to build the project:
```
   make
```
2. If successful, the following files will be created in the bin directory:
   - Executable: bin/MQLib
   - Library: bin/MQLib.a

---

#### Step 4: Verify the Installation

1. Test the executable by running:
```
   ./bin/MQLib
```

2. If the output confirms the library's functionality, the installation is complete.

---

### Troubleshooting

- make not found:
  Ensure make is installed and available in your PATH.

- Permission Denied:
  Ensure you have write permissions in the directory. Run the terminal as Administrator or use sudo if necessary.

- Compilation Errors:
  Check if all dependencies are installed and up-to-date.

---

### Additional Resources

For more information, visit the [MQLib GitHub Repository](https://github.com/MQLib/MQLib).
