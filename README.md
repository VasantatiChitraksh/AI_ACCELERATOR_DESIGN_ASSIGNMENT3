# AI Accelerator Design | Assignment 3

This repository contains the solution for **Assignment 3: Tensor Flattening and Reconstruction**. 
The goal of this project is to implement algorithms from scratch to convert multi-dimensional image tensors ($B \times C \times H \times W$) into flattened linear representations ($B \times (CHW)$) and back again, without using built-in library functions like `reshape()` or `flatten()`.

## Directory Structure

```
├── .gitignore          # Git ignore file (excludes venv and pycache)
├── README.md           # This readme file
├── report/
│   └── report.tex      # LaTeX source code for the assignment report
└── src/
    ├── main.py               # Main script running the evaluation metrics
    └── tensor_transform.py   # Core conversion and reconstruction algorithms
```

## Setup & Running

1. **Create and Activate Virtual Environment:**
   ```bash
   python -m venv venv
   # On Windows:
   .\venv\Scripts\activate
   # On macOS/Linux:
   source venv/bin/activate
   ```

2. **Install Dependencies:**
   ```bash
   pip install numpy
   ```

3. **Run Experiments:**
   ```bash
   python src/main.py
   ```

This will run the flattening and reconstruction algorithms on various tensor configurations (including MNIST, CIFAR, and synthetic feature maps) and print the Maximum Absolute Error (MAE) and Mean Absolute Error.