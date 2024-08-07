"""
Apply pruning, knowledge distillation and quantization to a model
""" 


# Imports
import wandb
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_pruning as tp
import os
import copy
import random
import numpy as np
from torch.optim import SGD
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from functools import partial
import logging
from datetime import datetime
from torch2trt import torch2trt
from pytorch_bench import get_model_size


assert torch.cuda.is_available(), "CUDA is not available!"

device = torch.device("cuda")

def setup_logging(base_path):
    os.makedirs(base_path, exist_ok=True)
    logging.basicConfig(filename=f"{base_path}/log.txt", level=logging.INFO)
    run_id = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    logging.info(f"Run ID: {run_id}")
    return run_id

@torch.no_grad()
def evaluate(model: nn.Module, dataloader: DataLoader, verbose=True) -> tuple:
    model.eval()
    num_samples = num_correct = loss = 0

    for inputs, targets in tqdm(dataloader, desc="eval", leave=False, disable=not verbose):
        inputs, targets = inputs.to(device), targets.to(device)
        outputs = model(inputs)
        loss += F.cross_entropy(outputs, targets, reduction="sum")
        num_samples += targets.size(0)
        num_correct += (outputs.argmax(dim=1) == targets).sum()

    return (num_correct / num_samples * 100).item(), (loss / num_samples).item()

def train(model: nn.Module, train_loader: DataLoader, test_loader: DataLoader, 
          epochs: int, lr: float, weight_decay=5e-4, callbacks=None, save=None, 
          save_only_state_dict=False) -> None:
    
    optimizer = SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=weight_decay)
    scheduler = CosineAnnealingLR(optimizer, epochs)
    criterion = nn.CrossEntropyLoss()
    best_acc = -1
    best_checkpoint = {}

    for epoch in range(epochs):
        model.train()
        for inputs, targets in tqdm(train_loader, leave=False):
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            if callbacks:
                for callback in callbacks:
                    callback()

        acc, val_loss = evaluate(model, test_loader)
        logging.info(
            f'Epoch {epoch + 1}/{epochs} | Val acc: {acc:.2f} | Val loss: {val_loss:.4f} | LR: {optimizer.param_groups[0]["lr"]:.6f}')
        
        if wandb.run:
            wandb.log({"val_acc": acc, "val_loss": val_loss, "lr": optimizer.param_groups[0]["lr"]})

        if acc > best_acc:
            best_checkpoint['state_dict'] = copy.deepcopy(model.state_dict())
            best_acc = acc
        scheduler.step()

    model.load_state_dict(best_checkpoint['state_dict'])
    if save:
        torch.save(model.state_dict() if save_only_state_dict else model, save)
    logging.info(f'Best val acc: {best_acc:.2f}')


def get_pruner(model, example_input, num_classes):
    imp = tp.importance.GroupNormImportance(p=2)
    pruner_entry = partial(tp.pruner.GroupNormPruner, isomorphic=True, global_pruning=True)

    ignored_layers = [m for m in model.modules() if 
                      (isinstance(m, nn.Linear) and m.out_features == num_classes) or
                      (isinstance(m, nn.modules.conv._ConvNd) and m.out_channels == num_classes)]

    return pruner_entry(
        model,
        example_input,
        importance=imp,
        iterative_steps=400,
        ch_sparsity=1.0,
        ignored_layers=ignored_layers,
    )

def progressive_pruning(pruner, model, target_value, example_inputs, mode='speedup'):
    model.eval()
    base_ops, base_params = tp.utils.count_ops_and_params(model, example_inputs=example_inputs)
    current_value = 1

    while current_value < target_value:
        pruner.step(interactive=False)
        pruned_ops, pruned_params = tp.utils.count_ops_and_params(model, example_inputs=example_inputs)
        current_value = float(base_ops) / pruned_ops if mode == 'speedup' else float(base_params) / pruned_params

    return current_value




def train_kd(model_student: nn.Module, model_teacher: nn.Module, train_loader: DataLoader, 
             test_loader: DataLoader, epochs: int, lr: float, temperature: float, alpha: float, 
             weight_decay=5e-4, callbacks=None, save=None, save_only_state_dict=False) -> None:
    optimizer = SGD(model_student.parameters(), lr=lr, momentum=0.9, weight_decay=weight_decay)
    scheduler = CosineAnnealingLR(optimizer, epochs)
    criterion = nn.CrossEntropyLoss()
    best_acc = -1
    best_checkpoint = {}

    for epoch in range(epochs):
        model_student.train()
        model_teacher.eval()
        for inputs, targets in tqdm(train_loader, leave=False):
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            out_student = model_student(inputs)
            out_teacher = model_teacher(inputs)

            predict_student = F.log_softmax(out_student / temperature, dim=1)
            predict_teacher = F.softmax(out_teacher / temperature, dim=1)
            loss = (nn.KLDivLoss(reduction="batchmean")(predict_student, predict_teacher) * 
                    (alpha * temperature * temperature) + 
                    criterion(out_student, targets) * (1-alpha))
            
            loss.backward()
            optimizer.step()

            if callbacks:
                for callback in callbacks:
                    callback()

        acc, val_loss = evaluate(model_student, test_loader)
        logging.info(
            f'KD - Epoch {epoch + 1}/{epochs} | Val acc: {acc:.2f} | Val loss: {val_loss:.4f} | LR: {optimizer.param_groups[0]["lr"]:.6f}')

        if wandb.run:
            wandb.log({"val_acc": acc, "val_loss": val_loss, "lr": optimizer.param_groups[0]["lr"]})
            
        if acc > best_acc:
            best_checkpoint['state_dict'] = copy.deepcopy(model_student.state_dict())
            best_acc = acc
        scheduler.step()

    model_student.load_state_dict(best_checkpoint['state_dict'])
    if save:
        torch.save(model_student.state_dict() if save_only_state_dict else model_student, save)
    logging.info(f'Best val acc after KD: {best_acc:.2f}')

def get_compression_params(compression_ratio):
    if compression_ratio <= 2:
        return compression_ratio, 32
    elif compression_ratio <= 4:
        return compression_ratio/2, 16
    else:
        return compression_ratio/4, 8




def apply_pruning_and_kd(model, train_loader, test_loader, example_input, num_classes, 
                         epochs=120, lr=0.01, temperature=4, alpha=0.9, 
                         compression_ratio=None, speed_up=None, random_seed=42):
    # Setup
    base_path = "results_experiments"
    run_id = setup_logging(base_path)
    logging.info(f"Model: {model.__class__.__name__}")
    
    # Set random seed
    set_random_seed(random_seed)
    
    # Prepare models
    model = model.to(device)
    example_input = example_input.to(device)
    model_teacher = copy.deepcopy(model)
    
    # Initial evaluation
    start_macs, start_params = tp.utils.count_ops_and_params(model, example_input)
    start_acc, start_loss = evaluate(model, test_loader)
    log_model_stats("Initial Model", start_macs, start_params, start_acc, start_loss)
    
    # Pruning
    pruner = get_pruner(model, example_input, num_classes)
    progressive_pruning(pruner, model, compression_ratio, example_input, 
                        mode='speedup' if speed_up else 'compression')
    # Knowledge Distillation
    logging.info('Starting Knowledge Distillation')
    model = train_kd(model, model_teacher, train_loader, test_loader, epochs, lr, temperature, alpha,
                     save=f'{base_path}/{run_id}/kd_model.pth')
    
    # Final evaluation
    end_macs, end_params = tp.utils.count_ops_and_params(model, example_input)
    end_acc, end_loss = evaluate(model, test_loader)
    log_model_stats("Final Model", end_macs, end_params, end_acc, end_loss)
    
    # Save model
    torch.save(model.state_dict(), f'{base_path}/{run_id}/final_model.pth')
    
    return model, end_macs, end_params, end_acc, end_loss

def apply_quantization(model, example_input, train_loader, bitwidth=16):
    logging.info('----- Quantization -----')
    torch.cuda.empty_cache()
    
    if bitwidth == 8:
        logging.info('Calibrating on train dataset...')
        calib_dataset = get_calibration_dataset(train_loader)
        model_trt = torch2trt(model, [example_input], fp16_mode=True, int8_mode=True, 
                              int8_calib_dataset=calib_dataset, max_batch_size=128)
        compression_ratio_quant = 4
    elif bitwidth == 16:
        model_trt = torch2trt(model, [example_input], fp16_mode=True, max_batch_size=128)
        compression_ratio_quant = 2
    else:
        model_trt = torch2trt(model, [example_input], max_batch_size=128)
        bitwidth = 32
        compression_ratio_quant = 1
    
    logging.info(f"Bit width: {bitwidth}")
    return model_trt, compression_ratio_quant

def optimize(model, train_loader, test_loader, example_input, num_classes,
             epochs=120, lr=0.01, temperature=4, alpha=0.9, 
             compression_ratio=2, speed_up=None, bitwidth=16, 
             wandb_project=None, random_seed=42):
    
    # Setup
    base_path = setup_experiment_folder()
    setup_logging(base_path)
    setup_wandb(wandb_project)
    
    # Initial model evaluation
    start_macs, start_params = tp.utils.count_ops_and_params(model, example_input)
    start_acc, start_loss = evaluate(model, test_loader)
    log_model_stats("Initial Model", start_macs, start_params, start_acc, start_loss)
    
    # Determine pruning strategy and quantization bitwidth
    compression_ratio, speed_up, bitwidth = determine_optimization_strategy(compression_ratio, speed_up)
    
    # Apply pruning and knowledge distillation
    model, end_macs, end_params, end_acc, end_loss = apply_pruning_and_kd(
        model, train_loader, test_loader, example_input, num_classes,
        epochs, lr, temperature, alpha, compression_ratio, speed_up, random_seed
    )
    
    # Apply quantization
    model_trt, compression_ratio_quant = apply_quantization(model, example_input, train_loader, bitwidth)
    
    # Log final results
    final_compression_ratio = (start_params / end_params) * compression_ratio_quant
    logging.info(f"Final Compression Ratio: {final_compression_ratio:.2f}")
    
    # Save final model
    torch.save(model_trt.state_dict(), f'{base_path}/model_trt.pth')
    
    # Log results to wandb if enabled
    if wandb_project:
        log_wandb_results(start_macs, start_params, start_acc, start_loss,
                          end_macs, end_params, end_acc, end_loss,
                          final_compression_ratio, bitwidth)
    
    return model_trt

# Helper functions
def setup_experiment_folder():
    base_path = f"results_experiments/{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
    os.makedirs(base_path, exist_ok=True)
    return base_path

def setup_logging(base_path):
    logging.basicConfig(filename=f"{base_path}/log.txt", level=logging.INFO)
    logging.info(f"Run ID: {os.path.basename(base_path)}")

def setup_wandb(wandb_project):
    if wandb_project:
        wandb.init(project=wandb_project)
        logging.info(f"Wandb initialized. Project: {wandb_project}")

def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

def log_model_stats(stage, macs, params, acc, loss):
    logging.info(f' ----- {stage}: -----')
    logging.info(f'Number of MACs = {macs/1e6:.3f} M')
    logging.info(f'Number of Parameters = {params/1e6:.3f} M')
    logging.info(f'Accuracy = {acc:.2f} %')
    logging.info(f'Loss = {loss:.3f}')
    logging.info(' ---------------------------')

def determine_optimization_strategy(compression_ratio, speed_up):
    compression_ratio, bitwidth = get_compression_params(compression_ratio or speed_up)
    return compression_ratio, speed_up, bitwidth

def get_calibration_dataset(train_loader):
    calib_dataset = []
    for i, (img, _) in enumerate(train_loader):
        calib_dataset.extend(img)
        if i == 2000:
            break
    return calib_dataset

def log_wandb_results(start_macs, start_params, start_acc, start_loss,
                      end_macs, end_params, end_acc, end_loss,
                      final_compression_ratio, bitwidth):
    wandb.run.summary.update({
        "start_macs (M)": f'{start_macs/1e6:.3f}',
        "start_params (M)": f'{start_params/1e6:.3f}',
        "start_acc (%)": f'{start_acc:.2f}',
        "start_loss": f'{start_loss:.3f}',
        "end_macs (M)": end_macs/1e6,
        "end_params (M)": end_params/1e6,
        "best_acc": end_acc,
        "best_loss": end_loss,
        "final_compression_ratio": final_compression_ratio,
        "bitwidth": bitwidth,
        "size (MB)": get_model_size(model)/8e6
    })