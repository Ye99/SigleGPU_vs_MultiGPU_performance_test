# Accelerate Performance Benchmark

This project demonstrates the performance benefits of using Hugging Face's Accelerate library for distributed training with PyTorch.

## Setting Up Accelerate

### Installation

1. **Create a conda environment:**
   ```bash
   conda create -n accelerate python=3.11 -y
   conda activate accelerate
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

### Configuration

Before running distributed training with Accelerate, you need to configure it:

1. **Run the configuration wizard:**
   ```bash
   accelerate config
   ```
   
   This interactive wizard will ask you several questions:
   - What compute environment you're using (local, remote, etc.)
   - How many machines to use
   - Mixed precision options (fp16, bf16, etc.)
   - Whether to use DeepSpeed or Fully Sharded Data Parallelism (FSDP)

2. **Quick auto-configuration:**
   ```bash
   # For single-machine, multi-GPU setup with mixed precision
   accelerate config default
   ```

3. **View your current configuration:**
   ```bash
   accelerate env
   ```

## Running the Benchmark

You can run the benchmark using the following commands:

```bash
# Basic single-process run
python main.py --dataset-size 5000 --batch-size 64 --epochs 2

# Multi-GPU run with Accelerate
accelerate launch main.py --dataset-size 5000 --batch-size 64 --epochs 2
```

## Performance Comparison

### Regular Python (`python main.py`):
- Training time: 37.26 seconds
- Throughput: 268.37 examples/second
- Used 1 GPU with batch size of 64

### Accelerate Launch (`accelerate launch main.py`):
- Training time: 22.33 seconds (40% faster!)
- Throughput: 447.78 examples/second (67% higher!)
- Used 2 GPUs with effective batch size of 128 (64 per GPU)

## Why Accelerate Is Now Faster

1. **Multi-GPU Utilization**: Accelerate automatically used both RTX 4060 Ti GPUs, while the regular Python run only used one GPU.

2. **Higher Effective Batch Size**: With accelerate, the batch size effectively doubles (64 per GPU × 2 GPUs = 128 total).

3. **Workload Size**: With 5,000 examples, there's enough work to distribute that the overhead of setting up distributed training is offset by the parallel processing benefits.

## Real-World Implications

In real-world scenarios, the benefits become even more dramatic as you:

1. Use larger datasets (tens of thousands to millions of examples)
2. Train more complex models
3. Run for more epochs
4. Use more GPUs

## When to Use Each Approach

- **Development/Debugging**: Regular `python main.py` for quick iteration
- **Production/Training**: `accelerate launch main.py` for maximum performance

## Key Takeaway

With a realistic workload, the benefits of Accelerate are clear - nearly higher throughput when using multiple GPUs!

## Requirements

See `requirements.txt` file for the complete list of dependencies.

## Cursor Rules

This project includes Cursor rules for consistent code formatting and best practices. The rules are available in the `.cursor/rules` directory via a symbolic link to the CursorRules submodule.

### Available Rules

- `python.mdc`: Python coding rules for consistent Python code formatting and best practices
- `frontend.mdc`: Frontend development rules for web development

### Setting Up the Rules

The rules come from the [CursorRules](https://github.com/Ye99/CursorRules) repository, which is included as a Git submodule. To initialize the submodule:

1. **If you're cloning the repository for the first time:**
   ```bash
   git clone --recurse-submodules https://github.com/yourusername/accelerate.git
   ```

2. **If you've already cloned the repository without the submodule:**
   ```bash
   git submodule init
   git submodule update
   ```

3. **To update the submodule to the latest version:**
   ```bash
   git submodule update --remote
   ```