# script.py

from accelerate import Accelerator
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from torch.utils.data import DataLoader, Dataset
import time
import argparse
import os

# Create a larger synthetic dataset for better demonstration
class SyntheticDataset(Dataset):
    def __init__(self, tokenizer, size=1000, seq_length=128):
        self.size = size
        # Pre-generate all the input_ids, attention_masks, and labels
        self.input_ids = torch.randint(0, tokenizer.vocab_size, (size, seq_length))
        self.attention_mask = torch.ones((size, seq_length))
        self.labels = torch.randint(0, 2, (size,))

    def __getitem__(self, idx):
        return {
            "input_ids": self.input_ids[idx],
            "attention_mask": self.attention_mask[idx],
            "labels": self.labels[idx]
        }

    def __len__(self):
        return self.size

def benchmark_training(accelerator, model, optimizer, dataloader, num_epochs=3):
    # Use Accelerator to prepare the model, optimizer, and dataloader
    model, optimizer, dataloader = accelerator.prepare(model, optimizer, dataloader)
    
    # Training loop with timing
    model.train()
    start_time = time.time()
    
    for epoch in range(num_epochs):
        epoch_start = time.time()
        running_loss = 0.0
        batch_count = 0
        
        for batch in dataloader:
            outputs = model(**batch)
            loss = outputs.loss
            accelerator.backward(loss)
            optimizer.step()
            optimizer.zero_grad()
            running_loss += loss.item()
            batch_count += 1
        
        # Only print from the main process
        if accelerator.is_main_process:
            avg_loss = running_loss / batch_count
            epoch_time = time.time() - epoch_start
            print(f"Epoch {epoch}, Loss: {avg_loss:.4f}, Time: {epoch_time:.2f}s")
    
    total_time = time.time() - start_time
    if accelerator.is_main_process:
        print(f"Total training time: {total_time:.2f}s")
    
    # Properly cleanup resources
    accelerator.wait_for_everyone()
    accelerator.free_memory()
    
    # If distributed training is used, properly destroy the process group
    if accelerator.distributed_type != "NO" and torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()
    
    return total_time

def main():
    parser = argparse.ArgumentParser(description="Benchmark Accelerate vs. Regular PyTorch")
    parser.add_argument("--dataset-size", type=int, default=10000, help="Number of examples in synthetic dataset")
    parser.add_argument("--batch-size", type=int, default=32, help="Training batch size")
    parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs")
    args = parser.parse_args()
    
    # Initialize Accelerator, which handles device placement and multi-GPU distribution
    accelerator = Accelerator()
    
    # Print environment info
    if accelerator.is_main_process:
        print("\nTraining Environment:")
        print(f"Number of processes: {accelerator.num_processes}")
        print(f"Distributed type: {accelerator.distributed_type}")
        print(f"Mixed precision: {accelerator.mixed_precision}")
        print(f"Dataset size: {args.dataset_size}")
        print(f"Batch size: {args.batch_size} (effective: {args.batch_size * accelerator.num_processes})")
        print(f"Using device: {accelerator.device}")
        if torch.cuda.is_available():
            print(f"CUDA device count: {torch.cuda.device_count()}")
            for i in range(torch.cuda.device_count()):
                print(f"  Device {i}: {torch.cuda.get_device_name(i)}")
    
    # Load model and tokenizer
    model_name = "distilbert-base-uncased"
    if accelerator.is_main_process:
        print(f"\nLoading model: {model_name}")
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)
    
    # Create synthetic dataset and dataloader
    if accelerator.is_main_process:
        print(f"Creating synthetic dataset with {args.dataset_size} examples")
    
    dataset = SyntheticDataset(tokenizer, size=args.dataset_size)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    
    # Define optimizer - using PyTorch's AdamW to avoid warning
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)
    
    # Run benchmarking
    if accelerator.is_main_process:
        print("\nStarting training...")
    
    total_time = benchmark_training(
        accelerator, model, optimizer, dataloader, num_epochs=args.epochs
    )
    
    # Summary
    if accelerator.is_main_process:
        print("\nPerformance Summary:")
        print(f"Dataset size: {args.dataset_size}")
        print(f"Batch size: {args.batch_size} (effective: {args.batch_size * accelerator.num_processes})")
        print(f"Epochs: {args.epochs}")
        print(f"Total training time: {total_time:.2f}s")
        
        examples_per_second = (args.dataset_size * args.epochs) / total_time
        print(f"Throughput: {examples_per_second:.2f} examples/second")
        
        if accelerator.num_processes > 1:
            print("\nTo compare with single-process performance, run without 'accelerate launch'")

if __name__ == "__main__":
    main()
