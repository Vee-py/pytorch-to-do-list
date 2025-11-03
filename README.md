# pytorch-tensors-to-do-list
🧠 What is a PyTorch Tensor?

A tensor in PyTorch is a multi-dimensional array, similar to:

a scalar (single number),

a vector (1D array),

a matrix (2D array),

or higher-dimensional data (3D, 4D, …).

Essentially, a tensor is like a NumPy array, but it’s optimized for:

GPU acceleration (using CUDA for fast computations),

automatic differentiation (used in training neural networks).

⚙️ Where is it used?

Tensors are the core data structure in PyTorch, used for:

Storing data (like input images, sound waves, text embeddings, etc.)

Performing mathematical operations (like matrix multiplication, convolution, etc.)

Building and training deep learning models

They’re used both in research and production for machine learning and AI applications.

🌍 Real-World Examples of PyTorch Tensor Use
🧩 1. Computer Vision

Self-driving cars — process camera images with tensors (each image is a 3D tensor: height × width × color channels)

Medical imaging — detect tumors in MRI or CT scans

Face recognition — encode facial features as tensors for identification
Use Case	Requirements
Learning / Small projects	CPU-only fine, 4–8GB RAM
Deep learning / Big models	GPU with ≥8GB VRAM, CUDA 12+
Apple M1/M2/M3	Works natively with Metal
Linux/Windows/macOS	All supported
🧩 1. Check Your System

First, confirm what type of system you’re using:

You Have	Best Setup
💻 Windows/Linux + NVIDIA GPU	Install PyTorch with CUDA for GPU acceleration
🍎 Mac (M1/M2/M3)	Use Metal (MPS) backend — built into PyTorch
🧠 No GPU / Older system	Use CPU-only version of PyTorch
⚙️ 2. Basic Requirements
Requirement	Details
Python	Version 3.8 – 3.12
Pip or Conda	For installing packages
Internet connection	To download dependencies
Admin rights (optional)	For system-wide installs
🐍 3. Recommended: Set Up a Virtual Environment

This helps avoid conflicts with other Python libraries.

🪟 On Windows:
python -m venv pytorch_env
pytorch_env\Scripts\activate

🐧 On Linux/Mac:
python3 -m venv pytorch_env
source pytorch_env/bin/activate


Now your terminal prompt should show (pytorch_env) — meaning the environment is active.

🔧 4. Install PyTorch

Go to the official PyTorch Get Started page
 (it auto-detects your OS),
or follow one of these ready-made commands 👇

🧠 A. CPU-only version (works on all systems)
pip install torch torchvision torchaudio

⚡ B. GPU version (NVIDIA CUDA support)

Make sure you have:

NVIDIA driver installed

CUDA 12.x or 11.x compatible with your GPU

Then run:

pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121


(Replace cu121 with your CUDA version, e.g., cu118)

Check CUDA availability:

import torch
print(torch.cuda.is_available())  # True = GPU ready

🍎 C. macOS (M1/M2/M3 Apple Silicon)

PyTorch now supports Apple’s Metal (MPS) backend automatically.

Install with:

pip install torch torchvision torchaudio


Then verify MPS works:

import torch
print(torch.backends.mps.is_available())  # True if GPU acceleration enabled

🧰 5. Verify Installation

Run Python in your terminal:

python


Then test:

import torch
x = torch.rand(2, 3)
print(x)
print("PyTorch version:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())


If it prints a tensor and no errors → ✅ PyTorch is installed correctly.

🧱 6. Optional but Helpful Tools
Tool	Command	Purpose
Jupyter Notebook	pip install notebook	Run interactive code
Anaconda	Download
	Easiest environment manager
VS Code	Download
	Great IDE for PyTorch
TorchVision	Included	Image datasets/models
TorchAudio	Included	Audio datasets/models
TorchText	pip install torchtext	NLP datasets/models
💡 7. Optional (Cloud Option)

If your computer is low on RAM or GPU:

Use Google Colab → colab.research.google.com

It has PyTorch preinstalled and gives you free GPU access.

✅ Quick Summary
Component	Recommendation
Python	3.10+
RAM	≥ 8 GB
GPU (optional)	NVIDIA CUDA 12+ or Apple MPS
Install command	pip install torch torchvision torchaudio
Verification	import torch; print(torch.__version__)

#start off
Simple analogy:

A scalar is a single number (0D tensor)
A vector is a list of numbers (1D tensor)
A matrix is a table of numbers (2D tensor)
A tensor is... any of the above, or higher dimensional!

Creating Tensors
pythonimport torch

# From Python lists
x = torch.tensor([1, 2, 3])
print(x)  # tensor([1, 2, 3])

# 2D tensor (matrix)
matrix = torch.tensor([[1, 2], [3, 4], [5, 6]])
print(matrix.shape)  # torch.Size([3, 2]) - 3 rows, 2 columns

# Common initialization methods
zeros = torch.zeros(3, 4)      # 3x4 tensor of zeros
ones = torch.ones(2, 3)         # 2x3 tensor of ones
random = torch.rand(2, 3)       # Random values between 0 and 1
randn = torch.randn(2, 3)       # Random values from normal distribution
arange = torch.arange(0, 10, 2) # tensor([0, 2, 4, 6, 8])
Tensor Attributes
Every tensor has important attributes:
pythonx = torch.randn(3, 4, 5)

print(x.shape)      # torch.Size([3, 4, 5]) - dimensions
print(x.dtype)      # torch.float32 - data type
print(x.device)     # cpu or cuda - where it's stored
print(x.requires_grad)  # False - whether to track gradients
Data Types
python# Specify data type
float_tensor = torch.tensor([1, 2, 3], dtype=torch.float32)
int_tensor = torch.tensor([1, 2, 3], dtype=torch.int64)

# Convert between types
x = torch.tensor([1.5, 2.7])
x_int = x.int()     # Convert to integer
x_long = x.long()   # Convert to long integer
Basic Operations
python# Element-wise operations
a = torch.tensor([1, 2, 3])
b = torch.tensor([4, 5, 6])

print(a + b)        # tensor([5, 7, 9])
print(a * b)        # tensor([4, 10, 18])
print(a ** 2)       # tensor([1, 4, 9])

# Matrix operations
A = torch.randn(3, 4)
B = torch.randn(4, 2)
C = torch.matmul(A, B)  # Matrix multiplication: (3,4) x (4,2) = (3,2)
# or use the @ operator
C = A @ B
Indexing and Slicing
pythonx = torch.tensor([[1, 2, 3],
                  [4, 5, 6],
                  [7, 8, 9]])

print(x[0])         # tensor([1, 2, 3]) - first row
print(x[:, 1])      # tensor([2, 5, 8]) - second column
print(x[0, 2])      # tensor(3) - element at row 0, col 2
print(x[1:, :2])    # Rows 1 onward, first 2 columns
Reshaping Tensors
pythonx = torch.arange(12)  # tensor([0, 1, 2, ..., 11])

# Reshape to 3x4
y = x.view(3, 4)
print(y.shape)  # torch.Size([3, 4])

# Reshape to 2x6
z = x.reshape(2, 6)

# Flatten
flat = y.view(-1)  # -1 means "infer this dimension"
print(flat.shape)  # torch.Size([12])

# Add/remove dimensions
unsqueezed = x.unsqueeze(0)  # Add dimension at position 0
print(unsqueezed.shape)  # torch.Size([1, 12])
GPU Acceleration
python# Check if GPU is available
print(torch.cuda.is_available())

# Move tensor to GPU
if torch.cuda.is_available():
    x = torch.randn(3, 4)
    x_gpu = x.to('cuda')  # or x.cuda()
    print(x_gpu.device)  # cuda:0
    
    # Move back to CPU
    x_cpu = x_gpu.to('cpu')  # or x_gpu.cpu()
Gradients (for Deep Learning)
python# Create tensor that tracks gradients
x = torch.tensor([2.0, 3.0], requires_grad=True)

# Perform operations
y = x ** 2
z = y.sum()

# Compute gradients
z.backward()  # Automatically compute dz/dx

print(x.grad)  # tensor([4., 6.]) - the gradients
Common Tensor Operations
python# Aggregations
x = torch.randn(3, 4)
print(x.sum())      # Sum all elements
print(x.mean())     # Average
print(x.max())      # Maximum value
print(x.min())      # Minimum value
print(x.argmax())   # Index of max value

# Along specific dimensions
print(x.sum(dim=0))   # Sum along rows (result: shape [4])
print(x.mean(dim=1))  # Mean along columns (result: shape [3])

# Concatenation
a = torch.ones(2, 3)
b = torch.zeros(2, 3)
c = torch.cat([a, b], dim=0)  # Stack vertically: shape [4, 3]
d = torch.cat([a, b], dim=1)  # Stack horizontally: shape [2, 6]
Broadcasting
PyTorch automatically expands tensors of different shapes:
pythonx = torch.ones(3, 4)
y = torch.tensor([1, 2, 3, 4])

# y is broadcast to match x's shape
z = x + y  # Works! y is treated as [[1, 2, 3, 4]] repeated 3 times
print(z.shape)  # torch.Size([3, 4])
In-place Operations
pythonx = torch.tensor([1, 2, 3])

# Regular operation (creates new tensor)
y = x + 5

# In-place operation (modifies x directly)
x.add_(5)  # Operations with _ suffix are in-place
print(x)   # tensor([6, 7, 8])
Key Takeaways

Tensors are the foundation - everything in PyTorch is a tensor
Shape matters - always be aware of your tensor dimensions
GPU acceleration - move tensors to GPU for faster computation
Automatic gradients - PyTorch tracks operations for backpropagation
Similar to NumPy - if you know NumPy, you'll feel at home

#common issues and fixes
🧩 1. Shape Mismatch Errors

Error Example:

RuntimeError: mat1 and mat2 shapes cannot be multiplied (2x3 and 2x2)

🔍 Cause:

You’re performing an operation like matrix multiplication or concatenation on tensors whose dimensions don’t align.

✅ Fix:

Check tensor shapes with .shape or .size() and adjust them using:

.view() or .reshape() to change shape

.transpose() or .permute() to reorder axes

.unsqueeze() / .squeeze() to add/remove dimensions

Example:

x = torch.rand(2, 3)
w = torch.rand(3, 4)
y = x @ w   # Works because (2x3) * (3x4) = (2x4)

⚙️ 2. Wrong Indexing or Out-of-Bounds Access

Error Example:

IndexError: index 3 is out of bounds for dimension 1 with size 3

🔍 Cause:

You’re trying to access a tensor element outside its valid range.

✅ Fix:

Check the tensor’s size before indexing:

x = torch.tensor([[1, 2, 3],
                  [4, 5, 6]])
print(x.shape)   # torch.Size([2, 3])
print(x[0, 2])   # OK
# print(x[0, 3]) -> ❌ IndexError

⚡ 3. Device Mismatch (CPU vs GPU)

Error Example:

RuntimeError: Expected all tensors to be on the same device, but found at least two devices, cuda:0 and cpu!

🔍 Cause:

Some tensors are on the CPU, others are on the GPU — PyTorch can’t mix them directly.

✅ Fix:

Move all tensors to the same device:

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
x = torch.rand(2, 2).to(device)
y = torch.rand(2, 2).to(device)
z = x + y

🧮 4. Dtype (Data Type) Mismatch

Error Example:

RuntimeError: expected scalar type Float but found Long

🔍 Cause:

Operations like addition or loss functions expect matching tensor types — e.g. float32 vs int64.

✅ Fix:

Convert using .to(dtype=...) or .float(), .long(), etc.

x = torch.tensor([1, 2, 3], dtype=torch.float32)
y = torch.tensor([1, 2, 3], dtype=torch.int64)
z = x + y.float()  # Convert y to float

Practice Tips
Start with these exercises:

Create tensors of different shapes and visualize them
Practice reshaping and slicing operations
Try basic mathematical operations
Experiment with moving tensors between CPU and GPU
Build simple computations and compute gradients
1. Official PyTorch Documentation

Source: PyTorch Tensors Documentation

The most authoritative and up-to-date reference.

Covers:

Tensor creation (torch.tensor, torch.arange, etc.)

Tensor operations (arithmetic, indexing, reshaping)

Device management (CPU, GPU)

Automatic differentiation (requires_grad)

Why use it: It’s written by the PyTorch developers and includes working examples.

🧠 2. PyTorch Tutorials

Source: PyTorch.org → Tutorials

Recommended ones:

“Tensors”: Deep Learning with PyTorch: A 60 Minute Blitz

“Tensor Basics” section introduces:

Tensor creation/manipulation

Operations

Interoperability with NumPy

Why use it: Great for hands-on learners — code + explanation side by side.

📗 3. Textbooks and Academic References
A. Deep Learning with PyTorch (by Eli Stevens, Luca Antiga, Thomas Viehmann)

Publisher: Manning Publications (2020)

ISBN: 9781617295263

Chapters 1–3 explain tensors, autograd, and computational graphs in a clear, visual way.

Includes real-world applications (image classification, NLP, GANs).

📖 Book link (Manning)

B. Deep Learning (Ian Goodfellow, Yoshua Bengio, Aaron Courville)

Publisher: MIT Press (2016)

Free online: https://www.deeplearningbook.org/

Chapter 2 ("Linear Algebra") and Chapter 3 ("Probability and Information Theory") explain tensors mathematically — beyond code.

Happy learning! 🚀
