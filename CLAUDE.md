# CLAUDE.md — AI Assistant Guide for 2026_Practical_Deep_learning_EIM12OB001

## Repository Overview

This is the repository for the **Practical Deep Learning** course at PSL University, Spring 2026. It contains weekly Jupyter notebook practicals that progressively build up deep learning fundamentals from scratch using NumPy, progressing toward full deep neural networks.

---

## Repository Structure

```
2026_Practical_Deep_learning_EIM12OB001/
├── README.md
├── CLAUDE.md                        # This file
├── environment.yml                  # Conda environment specification
├── requirements.txt                 # Pip requirements
├── week_1/
│   ├── vectorization_demo.py        # NumPy vectorization benchmark demo
│   └── practical_1/
│       ├── Python_Basics_with_Numpy.ipynb            # Student notebook
│       ├── Python_Basics_with_Numpy_Correction.ipynb # Reference solution
│       ├── public_tests.py          # Graded test functions
│       └── test_utils.py            # Test runner infrastructure
├── week_2/
│   └── practical_2/
│       ├── Logistic_Regression_with_a_Neural_Network_mindset.ipynb
│       ├── Logistic_Regression_with_a_Neural_Network_mindset_Correction.ipynb
│       ├── lr_utils.py              # Dataset loader (cat vs non-cat, HDF5)
│       └── public_tests.py
├── week_3/
│   └── practical_3/
│       ├── Planar_data_classification_with_one_hidden_layer.ipynb
│       ├── Planar_data_classification_with_one_hidden_layer_Correction.ipynb
│       ├── planar_utils.py          # Planar dataset generator and plotting utils
│       ├── public_tests.py
│       ├── testCases_v2.py          # Additional test cases
│       └── test_utils.py
└── week_4/
    ├── practical_4/
    │   ├── Building_your_Deep_Neural_Network_Step_by_Step.ipynb
    │   ├── dnn_utils.py             # Activation functions (sigmoid, relu) and backward passes
    │   ├── public_tests.py
    │   ├── testCases.py
    │   └── test_utils.py
    └── practical_5/
        ├── Deep Neural Network_Application.ipynb
        ├── dnn_app_utils_v3.py      # Full DNN utilities (forward/backward, predict, visualization)
        └── public_tests.py
```

---

## Curriculum Progression

| Week | Practical | Topic | Key Concepts |
|------|-----------|-------|--------------|
| 1 | 1 | Python & NumPy Basics | Sigmoid, derivatives, vectorization, L1/L2 loss |
| 2 | 2 | Logistic Regression as Neural Network | Forward/backward prop, gradient descent, cat classifier |
| 3 | 3 | Shallow Neural Network | 1-hidden-layer NN, tanh activation, planar datasets |
| 4 | 4 | Deep Neural Network (Step-by-Step) | L-layer forward/backward, ReLU, sigmoid, caches |
| 4 | 5 | Deep Neural Network (Application) | Full DNN applied to cat vs non-cat image classification |

---

## Environment Setup

### Conda (recommended)

```bash
conda env create -f environment.yml
conda activate 2026_Practical_Deep_learning_EIM12OB001
```

### Pip

```bash
pip install -r requirements.txt
```

### Key Dependencies

- **Python 3.10**
- **numpy** — all core math and array operations
- **matplotlib / seaborn** — visualization
- **scikit-learn** — utility datasets (planar data)
- **h5py** — loading HDF5 datasets (cat vs non-cat images)
- **pytorch** — available but not used in early practicals
- **tensorflow / keras** — available but not used in early practicals
- **jupyterlab** — notebook interface
- **kagglehub, dlai-tools** — DeepLearning.AI integration tools
- **music21, mido, pydub** — audio (for later practicals)

---

## Development Workflow

### Running Notebooks

```bash
jupyter lab
```

Open the desired `.ipynb` file from the weekly folder. Each notebook contains:
1. Explanatory markdown cells
2. `### START CODE HERE ###` / `### END CODE HERE ###` blocks for student implementations
3. In-notebook test calls using `public_tests.py` functions

### Running Tests

Tests are invoked from within the notebook cells using the public test functions. Each `public_tests.py` exports named test functions (e.g., `basic_sigmoid_test(target)`) that take the student's function as argument.

Test runner in `test_utils.py` (weeks 1, 3) uses a case-based framework:

```python
# Example usage within notebooks
from public_tests import basic_sigmoid_test
basic_sigmoid_test(basic_sigmoid)
```

Test runner in `test_utils.py` (weeks 4–5) provides `single_test` and `multiple_test` helpers.

**Test case types:**
- `datatype_check` — verifies return type (e.g., `np.ndarray`, `float`)
- `shape_check` — verifies array shape
- `equation_output_check` — uses `np.allclose()` for numerical comparison

### Checking Vectorization Performance

```bash
python week_1/vectorization_demo.py
```

Compares `np.dot` vectorized computation vs explicit Python loop.

---

## Code Conventions

### NumPy Array Shape Conventions

All practicals follow strict column-vector conventions inherited from DeepLearning.AI:

- **Input matrix X**: shape `(n_x, m)` — features as rows, examples as columns
- **Labels Y**: shape `(1, m)` — row vector
- **Weight matrices W**: shape `(n_current, n_prev)`
- **Bias vectors b**: shape `(n_current, 1)` — always a column vector, not a scalar
- **Activations A**: shape `(n_current, m)`

### Parameter Dictionary Keys

Parameters for L-layer networks are stored in dictionaries:
```python
parameters = {
    "W1": ...,  # shape (n_1, n_0)
    "b1": ...,  # shape (n_1, 1)
    "W2": ...,  # shape (n_2, n_1)
    "b2": ...,  # shape (n_2, 1)
    # ... up to layer L
}
```

Gradients follow the same naming with `d` prefix:
```python
grads = {"dW1": ..., "db1": ..., "dW2": ..., "db2": ...}
```

### Cache Convention (Week 4+)

Forward propagation stores caches for backprop:
- **Linear cache**: `(A_prev, W, b)`
- **Activation cache**: `Z` (pre-activation value)
- **Combined cache**: `(linear_cache, activation_cache)`
- **All caches**: list of per-layer caches, indexed `0` to `L-1`

### Random Seeds

All test cases set explicit random seeds (`np.random.seed(N)`) for reproducibility. Student implementations **must not** set their own seeds inside functions.

### No Global Variables in Functions

The test framework explicitly checks that functions do not rely on global variables. All inputs must be passed as arguments.

---

## Utility Files Reference

### `week_1/practical_1/test_utils.py`

Provides `test(test_cases, target)` — iterates over test cases, checking type, shape, and value.

### `week_2/practical_2/lr_utils.py`

```python
load_dataset()  # Returns train_x, train_y, test_x, test_y, classes from HDF5 files
```
Expects HDF5 files at `datasets/train_catvnoncat.h5` and `datasets/test_catvnoncat.h5` relative to the notebook directory.

### `week_3/practical_3/planar_utils.py`

```python
load_planar_dataset()    # Flower-shaped 2D binary classification dataset (400 examples)
load_extra_datasets()    # Noisy circles, moons, blobs, Gaussian quantiles
plot_decision_boundary(model, X, y)  # Visualize classifier boundary
sigmoid(x)               # Scalar/array sigmoid
```

### `week_4/practical_4/dnn_utils.py`

```python
sigmoid(Z)           # Returns (A, cache) where cache=Z
relu(Z)              # Returns (A, cache) where cache=Z
relu_backward(dA, cache)     # Returns dZ
sigmoid_backward(dA, cache)  # Returns dZ
```

### `week_4/practical_5/dnn_app_utils_v3.py`

Full reference implementation of all DNN building blocks plus:
```python
load_data()                       # Load cat vs non-cat HDF5 dataset
predict(X, y, parameters)         # Returns predictions, prints accuracy
print_mislabeled_images(...)       # Visualize misclassified examples
initialize_parameters(n_x, n_h, n_y)       # 2-layer init
initialize_parameters_deep(layer_dims)     # L-layer init
linear_forward(A, W, b)
linear_activation_forward(A_prev, W, b, activation)  # 'relu' or 'sigmoid'
L_model_forward(X, parameters)
compute_cost(AL, Y)
linear_backward(dZ, cache)
linear_activation_backward(dA, cache, activation)
L_model_backward(AL, Y, caches)
update_parameters(parameters, grads, learning_rate)
```

---

## File Naming Conventions

- Student notebooks: `<Topic_Name>.ipynb` (no suffix)
- Reference solutions: `<Topic_Name>_Correction.ipynb`
- Hidden files: `.hidden` and `.hidden.save` — contain grading metadata; do not modify
- Test files: `public_tests.py` (imported from notebooks), `test_utils.py` (test framework)
- Utility files: descriptive names like `lr_utils.py`, `dnn_utils.py`, `planar_utils.py`

---

## Important Constraints for AI Assistants

1. **Never modify `public_tests.py`** — these define the grading contract. If tests fail, fix the student implementation, not the tests.

2. **Never modify `_Correction.ipynb` files** — these are reference solutions. Students should implement functions in the non-correction notebooks.

3. **Never set `np.random.seed()` inside student functions** — seeds are set externally by tests to ensure reproducibility.

4. **Always return the exact types expected** — tests check `isinstance()`. Return `float` (not `np.float64`) where `float` is expected; return `np.ndarray` (not lists) where arrays are expected.

5. **Avoid using Python loops where vectorization is possible** — the course emphasizes NumPy vectorization as a core learning objective.

6. **Maintain column-vector conventions** — bias `b` must be shape `(n, 1)`, not `(n,)`. Use `keepdims=True` in reductions when needed.

7. **Do not use global variables inside functions** — the test framework will raise an `AssertionError` if implementations rely on globals.

8. **Dataset files are not in the repo** — HDF5 dataset files (`train_catvnoncat.h5`, `test_catvnoncat.h5`) must be downloaded separately and placed in a `datasets/` subdirectory relative to the notebook.

---

## Git Workflow

This repo follows a fork + branch model:

- `master` — main branch (upstream from `melodiemonod`)
- Feature/work branches prefixed with `claude/` for AI-assisted development

```bash
git checkout -b claude/<description>
git add <specific-files>
git commit -m "descriptive message"
git push -u origin claude/<description>
```

Merges from upstream:
```bash
git fetch origin master
git merge origin/master
```
