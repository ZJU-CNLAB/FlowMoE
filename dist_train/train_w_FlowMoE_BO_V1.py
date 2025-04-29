#!/usr/bin/env python3

# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import os
import torch
import torch.optim as optim
import torch.nn.functional as F
from torch import nn, Tensor
import argparse
import schemoe_custom_kernel
import torch.distributed as dist
import math
from contextlib import nullcontext
from typing import Any
import time
from torch.autograd import Function
import queue
import threading
import numpy as np
from scipy.special import erf
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern
from skopt import gp_minimize
from skopt.space import Real
from skopt.utils import use_named_args
from scipy.interpolate import Rbf, interp1d
import random

def writeFile_add(filename, data):
    file_handle = open(filename, mode='a')
    file_handle.write(data)
    file_handle.close()

def decorate_trace_handler(args, rank):
    def trace_handler(prof):
        print(prof.key_averages().table(sort_by="self_cuda_time_total", row_limit=-1))
        if rank == 0:
            prof.export_chrome_trace(
                "./batch_size"
                + str(args.batch_size)
                + "#num_tokens"
                + str(args.num_tokens)
                + "#model_dim"
                + str(args.model_dim)
                + "#hidden_size"
                + str(args.hidden_size)
                + "#num_local_experts"
                + str(args.num_local_experts)
                + "#capacity_factor"
                + str(args.capacity_factor)
                + "#a2a_ffn_overlap_degree"
                + str(args.a2a_ffn_overlap_degree)
                + "#step_num"
                + str(prof.step_num)
                + ".json"
            )

    return trace_handler


parser = argparse.ArgumentParser()

parser.add_argument("--local_rank", type=int, default=-1)
parser.add_argument("--batch_size", type=int, default=16)
parser.add_argument("--num_tokens", type=int, default=512)
parser.add_argument("--model_dim", type=int, default=2048)
parser.add_argument("--hidden_size", type=int, default=2048)
parser.add_argument("--num_local_experts", type=int, default=2)
parser.add_argument("--dtype", type=str, default="float32")
parser.add_argument("--fp32_gate", default=False, action="store_true")
parser.add_argument("--top", type=int, default=2)
parser.add_argument("--a2a_ffn_overlap_degree", type=int, default=1)
parser.add_argument("--num_steps", type=int, default=25)
parser.add_argument("--capacity_factor", type=float, default=1.0)
parser.add_argument("--parallel_type", type=str, default="auto")
parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
parser.add_argument("--use_2dh", default=False, action="store_true")
parser.add_argument("--record_shapes", default=False, action="store_true")
parser.add_argument("--with_stack", default=False, action="store_true")
parser.add_argument("--log", type=str, default="test.log")
parser.add_argument("--encode", type=str, default="no")

parser.add_argument("--num_heads", type=str, default=8)
parser.add_argument("--lr", type=str, default=0.01)
parser.add_argument("--model_name", type=str, default="GPT2-Tiny-MoE")

args = parser.parse_args()

dist.init_process_group("nccl")

dist_rank, dist_world_size = dist.get_rank(), dist.get_world_size()

args.local_rank = os.environ.get("LOCAL_RANK", 0)


def dist_print(*args):
    if dist_rank == 0:
        print(*args)


device = torch.device("cuda:%s" % args.local_rank)
torch.cuda.set_device(device)

torch.set_printoptions(sci_mode=False)

if args.dtype == "float32":
    torch.set_default_dtype(torch.float32)
elif args.dtype == "float64":
    torch.set_default_dtype(torch.float64)
elif args.dtype == "float16":
    torch.set_default_dtype(torch.float16)
elif args.dtype == "bfloat16":
    torch.set_default_dtype(torch.bfloat16)
else:
    raise Exception("Unrecognized data type specified: %s" % args.dtype)

from schemoe.impls import communicate as C

torch.manual_seed(0)

all_reduce_threads_group = []
a2a_is_run = False
# a2a_condition = threading.Condition()
all_reduce_lock = threading.Lock()

grad_block_size = 2 * 1024 * 1024

# Communication pool management
def comm_thread_fn(dist_world_size, grad_queue):
    """
    Communication thread: Perform All-Reduce, then divide by world_size.
    """

    torch.cuda.set_device(device)
    global a2a_is_run

    while True:
        try:
            grad_chunk = grad_queue.get(timeout=0.001)
        except queue.Empty:
            continue

        # Training end sign
        if grad_chunk is None:
            # print("Find None!", device)
            break

        # Parameter update sign for each iteration
        global update_flag
        if grad_chunk == "update_point":
            update_flag = True
            continue

        # Waiting for all-to-all communication to end
        while a2a_is_run == True:
            continue

        # all-reduce communication
        work = dist.all_reduce(grad_chunk, op=dist.ReduceOp.SUM, async_op=True)
        work.wait()

        grad_chunk /= dist_world_size
        # print(f"All-reduce completed for chunk of size {grad_chunk.numel()}.")


# communication operation
class CommOperation(Function):
    @staticmethod
    def forward(ctx, input):
        output = all_to_all(input)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        global a2a_is_run
        a2a_is_run = True

        grad_input = all_to_all(grad_output)

        a2a_is_run = False
        return grad_input


def all_to_all(input_split):
    """
    Custom all-to-all communication
    """

    input_split = input_split.contiguous()
    output_split = torch.zeros_like(input_split)

    dist.all_to_all_single(
        output_split,
        input_split,
        group=dist.group.WORLD
    )

    return output_split


class SimpleTransformerMoE(nn.Module):
    def __init__(self, model_dim, num_layers, num_heads, hidden_size, num_local_experts):
        super(SimpleTransformerMoE, self).__init__()

        # Define the Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(model_dim, num_heads, hidden_size, num_local_experts)
            for _ in range(num_layers)
        ])

    def forward(self, x, num_tokens, a2a_ffn_overlap_degree, args):
        for block in self.blocks:
            x = block(x, num_tokens, a2a_ffn_overlap_degree, args)
        return x


class TransformerBlock(nn.Module):
    def __init__(self, model_dim, num_heads, hidden_size, num_local_experts):
        super(TransformerBlock, self).__init__()
        self.multihead_attn = nn.MultiheadAttention(embed_dim=model_dim, num_heads=num_heads, batch_first=True)
        self.fc1_weight = nn.Parameter(torch.randn(num_local_experts * dist_world_size, model_dim, hidden_size,
                                                   dtype=torch.get_default_dtype(), device=device))
        self.fc2_weight = nn.Parameter(torch.randn(num_local_experts * dist_world_size, hidden_size, model_dim,
                                                   dtype=torch.get_default_dtype(), device=device))

    def forward(self, x, num_tokens, a2a_ffn_overlap_degree, args):
        input_split = self._split_input(x, a2a_ffn_overlap_degree)

        for i in range(a2a_ffn_overlap_degree):
            input_split[i], _ = self.multihead_attn(input_split[i], input_split[i], input_split[i])
            # input_split[i] = schemoe_custom_kernel.compress_operation(input_split[i], args.encode, "naive")
            input_split[i] = CommOperation.apply(input_split[i])

        for i in range(a2a_ffn_overlap_degree):
            # input_split[i] = schemoe_custom_kernel.decompress_operation(input_split[i])
            input_split[i] = torch.matmul(input_split[i], self.fc1_weight)
            input_split[i] = F.relu(input_split[i])
            input_split[i] = torch.matmul(input_split[i], self.fc2_weight)
            # input_split[i] = schemoe_custom_kernel.compress_operation(input_split[i], args.encode, "naive")
            input_split[i] = CommOperation.apply(input_split[i])

        # for i in range(a2a_ffn_overlap_degree):
        #     input_split[i] = schemoe_custom_kernel.decompress_operation(input_split[i])
        #     input_split[i] = input_split[i].view(input_size)

        output = torch.cat(input_split, dim=1)

        return output

    def _split_input(self, input, a2a_ffn_overlap_degree):
        split_size = input.shape[1] // a2a_ffn_overlap_degree
        input_split = list(input.split(split_size, dim=1))
        for i in range(a2a_ffn_overlap_degree):
            input_split[i] = input_split[i].contiguous()
        return input_split


def backward_hook(module, grad_in, grad_out, grad_queue):
    """
    This hook function will:
    1. Split the gradient into chunks (each chunk has a size of grad_block_size, e.g., 100KB).
    2. Store the chunks in a queue to be used for all-reduce.
    """

    grad = grad_in[0]  # Gradient in the first element of grad_in
    grad_size = grad.numel()
    grad_chunk_size = grad_block_size // grad.element_size()  # Calculate the number of elements per chunk

    # Ensure grad_chunk_size is valid
    if grad_chunk_size <= 0:
        raise ValueError(f"grad_block_size ({grad_block_size}) is too small for element size ({grad.element_size()})")

    # Split the gradient into smaller chunks
    num_chunks = (grad_size + grad_chunk_size - 1) // grad_chunk_size

    for i in range(num_chunks):
        start = i * grad_chunk_size
        end = min((i + 1) * grad_chunk_size, grad_size)
        grad_chunk = grad.reshape(-1)[start:end]

        grad_queue.put(grad_chunk)  # Put the chunk in the queue

    # Print for debugging
    # print(f"Gradient has been split into {num_chunks} chunks, each of size {grad_chunk_size * grad.element_size() / 1024:.2f} KB.")


def register_hooks(model, grad_queue):
    """
    Register hooks for layers in the model to access gradients and split them.
    """
    # Register backward hook to multihead attention layers
    for name, module in model.named_modules():
        if isinstance(module, nn.MultiheadAttention):
            module.register_full_backward_hook(
                lambda module, grad_in, grad_out: backward_hook(module, grad_in, grad_out, grad_queue))


update_flag = False


def train(
        num_layers,
        batch_size,
        num_tokens,
        model_dim,
        hidden_size,
        num_local_experts,
        top_value,
        a2a_ffn_overlap_degree,
        capacity_factor,
        num_heads,
):
    def zc(x, y):
        return (x + y - 1) // y * y

    expert_num = num_local_experts * dist_world_size
    x = torch.tensor(
        torch.randn(
            [
                expert_num,
                zc(
                    int(top_value * math.ceil(batch_size * num_tokens / expert_num) * capacity_factor),
                    a2a_ffn_overlap_degree if args.encode != "zfp" else a2a_ffn_overlap_degree * 4,
                ),
                model_dim,
            ],
            dtype=torch.float32,
            device="cpu",
        )
            .detach()
            .numpy(),
        dtype=torch.get_default_dtype(),
        requires_grad=False,
        device=device,
    ).requires_grad_(True)
    lst = []

    print(x.size())

    tuples = (
        dist_world_size,
        args.dtype,
        model_dim,
        hidden_size,
        batch_size * num_tokens,
        num_local_experts,
        top_value,
        a2a_ffn_overlap_degree,
        capacity_factor,
        device,
    )
    dist_print(
        "[Benchmark] world_size = %s, dtype = %s, model_dim = %s, hidden_size = %s, samples = %s, num_local_experts = %s, topK = %s, a2a_ffn_overlap_degree = %s, capacity_factor = `%s`, device = `%s`"
        % tuples
    )

    # Define the queue to hold gradient chunks
    grad_queue = queue.Queue()

    # grad_block_size_history[i], pre_iteration_history[i]
    max_history_size = 9
    grad_block_size_history = []
    pre_iteration_history = []

    # First init: set initial grad_block_size (e.g. 1MB = 1024*1024)
    global grad_block_size

    # Bayesian optimization parameters
    n_bo_iter = 8
    n_train_steps_each = 10  # For each BO iteration, train 10 steps and measure the averaging time
    acc_step = 0
    acc_time = 0

    global update_flag

    # Create an Transformer class
    transformer_moe = SimpleTransformerMoE(model_dim, num_layers, num_heads, hidden_size, num_local_experts).to(device)

    # Register hooks for gradient splitting
    register_hooks(transformer_moe, grad_queue)

    # optimizer = torch.optim.Adam(transformer_moe.parameters(), lr=args.lr)  # 使用Adam优化器
    optimizer = torch.optim.SGD(transformer_moe.parameters(), lr=0.0001)

    # warm start
    hot_steps = 5
    hot_time = 0

    # start warm_comm_thread
    warm_comm_thread = threading.Thread(
        target=comm_thread_fn,
        args=(dist_world_size, grad_queue),
        daemon=True
    )
    warm_comm_thread.start()

    for _ in range(hot_steps):
        schemoe_custom_kernel.clear_ptr_lst()
        optimizer.zero_grad()
        input = x.clone().requires_grad_(True)
        # forward
        output = transformer_moe(input, num_tokens, a2a_ffn_overlap_degree, args)
        # loss function
        loss = torch.mean((output - input) ** 2)
        # backward
        loss.backward()
        # wait all-reduce communication to end
        update_flag = False
        grad_queue.put("update_point")
        while True:
            if update_flag:
                break
        # parameter update
        optimizer.step()
        torch.cuda.synchronize()
        # torch.distributed.barrier()
        if dist_rank == 0:
            print("warm start step:", _)

    grad_queue.put(None)
    # wait warm_comm_thread to end
    warm_comm_thread.join()
    # print("Training finished.", dist_rank)

    # Using Bayesian optimization to predict near-optimal grad_block_size
    grad_block_size = 4 * 1024 * 1024
    # grad_block_size = random.choice([128 * 1024, 256 * 1024, 512 * 1024, 1024 * 1024, 2 * 1024 * 1024, 4 * 1024 * 1024, 8 * 1024 * 1024])

    for i in range(n_bo_iter):
        # grad_block_size = grad_block_size*2

        BO_start_time = time.time()

        # start BO_comm_thread
        BO_comm_thread = threading.Thread(
            target=comm_thread_fn,
            args=(dist_world_size, grad_queue),
            daemon=True
        )
        BO_comm_thread.start()

        for _ in range(n_train_steps_each):
            schemoe_custom_kernel.clear_ptr_lst()
            optimizer.zero_grad()
            input = x.clone().requires_grad_(True)
            # forward
            output = transformer_moe(input, num_tokens, a2a_ffn_overlap_degree, args)
            # loss function
            loss = torch.mean((output - input) ** 2)
            # backward
            loss.backward()
            # wait all-reduce communication to end
            update_flag = False
            grad_queue.put("update_point")
            while True:
                if update_flag:
                    break
            # parameter update
            optimizer.step()
            torch.cuda.synchronize()
            # torch.distributed.barrier()
            if dist_rank == 0:
                print("BO step:", i * n_bo_iter + _)

        grad_queue.put(None)
        # wait BO_comm_thread to end
        BO_comm_thread.join()
        # print("Training finished.", dist_rank)

        grad_block_size_history.append(grad_block_size)
        pre_iteration_history.append(((time.time() - BO_start_time) / n_train_steps_each))

        grad_block_size_history_np = np.array(grad_block_size_history)
        pre_iteration_history_np = np.array(pre_iteration_history)
        ###############################################################################################################
        if dist_rank == 0:
            if len(grad_block_size_history_np) < 4:
                next_grad_block_size = random.choice(
                    [128 * 1024, 256 * 1024, 512 * 1024, 1024 * 1024, 2 * 1024 * 1024, 4 * 1024 * 1024,
                     8 * 1024 * 1024])
                while next_grad_block_size in grad_block_size_history_np:
                    next_grad_block_size += 128 * 1024
            else:
                inter_func = interp1d(grad_block_size_history_np, pre_iteration_history_np, kind='cubic',
                                      fill_value='extrapolate', bounds_error=False)

                # inter_func = Rbf(grad_block_size_history, pre_iteration_history, function='gaussian')

                # Define an objective function to interpolate based on known data points
                def objective_function(x):
                    return float(inter_func([x]))

                # Defining the search space
                search_space = [Real(128 * 1024, 16 * 1024 * 1024, name='x')]

                # Optimization process
                result = gp_minimize(
                    func=objective_function,
                    dimensions=search_space,
                    acq_func="EI",
                    acq_optimizer="lbfgs",
                    x0=grad_block_size_history_np[:, None].tolist(),
                    y0=pre_iteration_history_np.tolist(),
                    n_calls=25,
                    xi=0.1,  # EI hyperparameter
                    noise=1e-10,
                    random_state=0
                )

                # Output result
                next_grad_block_size = round(result.x[0] / 1024) * 1024

                if i != n_bo_iter - 1:
                    tried = 0
                    while next_grad_block_size in grad_block_size_history_np and tried < 10:
                        next_grad_block_size += random.choice([-1, 1]) * 128 * 1024
                        next_grad_block_size = int(np.clip(next_grad_block_size, 128 * 1024, 16 * 1024 * 1024))
                        next_grad_block_size = round(next_grad_block_size / 1024) * 1024
                        tried += 1
                else:
                    print("near-optimal grad_block_size", next_grad_block_size)
                    print("predict time:", result.fun)


            # Broadcast
            bcast_tensor = torch.tensor([next_grad_block_size], dtype=torch.int64, device=device)
            dist.broadcast(bcast_tensor, src=0)

            grad_block_size = bcast_tensor[0].item()
            print("grad_block_size is updated:", grad_block_size)
            print("grad_block_size_history:", grad_block_size_history_np)
            print("pre_iteration_history:", pre_iteration_history_np)
        else:
            # rank != 0
            bcast_tensor = torch.tensor([-1], dtype=torch.int64, device=device)
            dist.broadcast(bcast_tensor, src=0)
            grad_block_size = bcast_tensor[0].item()
            print("grad_block_size is updated:", grad_block_size)
        ###############################################################################################################

    # grad_block_size_history = np.array(grad_block_size_history)
    # pre_iteration_history = np.array(pre_iteration_history)

    # grad_block_size = 64 * 1024
    # start Train_comm_thread
    Train_comm_thread = threading.Thread(
        target=comm_thread_fn,
        args=(dist_world_size, grad_queue),
        daemon=True
    )
    Train_comm_thread.start()

    iter_start_time = time.time()
    # with torch.no_grad():
    for _ in range(args.num_steps):
        schemoe_custom_kernel.clear_ptr_lst()
        optimizer.zero_grad()

        input = x.clone().requires_grad_(True)

        # forward
        output = transformer_moe(input, num_tokens, a2a_ffn_overlap_degree, args)

        # loss function
        loss = torch.mean((output - input) ** 2)

        # backward
        loss.backward()

        # wait all-reduce communication to end
        update_flag = False
        grad_queue.put("update_point")
        while True:
            if update_flag:
                break

        # parameter update
        optimizer.step()

        torch.cuda.synchronize()

        # torch.distributed.barrier()

        if dist_rank == 0:
            print("step:", _)

    grad_queue.put(None)
    # wait Train_comm_thread to end
    Train_comm_thread.join()
    # print("Training finished.", dist_rank)

    if dist_rank == 0:
        speed = batch_size * num_tokens * dist_world_size / ((time.time() - iter_start_time) / args.num_steps)
        print("average training speed:", speed, "tokens/s")
        per_iter_time = (time.time() - iter_start_time) / args.num_steps
        print("one iteration time:", per_iter_time * 1000, "ms")

        # res = 'batch_size: {} num_tokens: {} model_dim: {} hidden_size: {} capacity_factor: {} one_iteration_time: {}'.format(batch_size, num_tokens, model_dim, hidden_size, capacity_factor, per_iter_time * 1000) + '\n'
        # filename = './result/Customized_MoE_layers_FlowMoE.txt'
        # writeFile_add(filename, res)

# 512, 1024, 2048, 4096, 8192
for batch_size in [
    2, 4, 8,
]:
    for num_tokens in [
        512, 1024, 2048,
    ]:
        for model_dim in [
            512, 1024, 2048, 4096, 8192
        ]:
            for hidden_size in [
                512, 1024, 2048, 4096, 8192,
            ]:
                for num_local_experts in [
                    1,
                ]:
                    for top_value in [
                        2,
                    ]:
                        for capacity_factor in [
                            1.0, 1.1, 1.2
                        ]:
                            for num_heads in [
                                8,
                            ]:
                                train(
                                    1,
                                    batch_size,
                                    num_tokens,
                                    model_dim,
                                    hidden_size,
                                    num_local_experts,
                                    top_value,
                                    args.a2a_ffn_overlap_degree,
                                    capacity_factor,
                                    num_heads,
                                )
