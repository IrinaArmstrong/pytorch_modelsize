# Basic
import sys
import numpy as np
from typing import List
from collections import namedtuple

# Torch utils
import torch
import torch.nn as nn

import logging
logging.basicConfig(
    level=logging.DEBUG,
    format="%(levelname)s %(message)s",
    datefmt="%d/%b/%Y %H:%M:%S",
    stream=sys.stdout)

Parameter = namedtuple('Parameter', ['size', 'bits'],
                       defaults=[np.asarray((0, 0)), 32])


class SizeEstimator(object):

    def __init__(self, model: nn.Module, input_size: List[int],
                 input_n_bits: int = 32, cuda_is_available: bool = False):
        """
        Estimates the size of PyTorch models in memory for a given input size and data precision, measured in bits.
        So default input type of torch.float32 equals to 32 bits precision.
        ---
        As a special option, you can estimate the amount of memory occupied on the graphics accelerator (CUDA),
        if such is available.
        ! Note: In this case SizeEstimator monitors only memory occupied by tensors,
        not all memory slots that are used by caching memory allocator in PyTorch.
        """
        self._model = model
        self._input_size = input_size
        self._input_n_bits = input_n_bits
        # Note: current version of SizeEstimator works only with models which are stored on CPU
        # for GPU size estimation call `estimate_cuda_memory_usage()`
        self._model_on_cuda = False
        self._model_init_device_id = None

        # Calculate
        self.__check_device()
        self._parameters_sizes = self._get_parameter_sizes()
        self._output_sizes = self._get_output_sizes()
        self._parameters_bits = self._calculate_parameters_weight()
        self._forward_backward_bits = self._calculate_forward_backward_weight()
        self._input_weight = self._calculate_input_weight()
        self.__restore_device()

    def __check_device(self):
        """
        SizeEstimator works only with models which are stored on CPU, so if networks is allocated in CUDA memory,
        it will be temporary re-allocated on CPU.
        """
        params_device = all([param.get_device() == -1 for param in self._model.parameters()])
        if not params_device:  # some of model's parameters tensors resides on GPU
            self._model_on_cuda = True
            # select first device id, if model is allocated on multiple GPU's,
            # after size estimation it will be placed on single one (actually, random one)
            self._model_init_device_id = np.unique([param.get_device() for param in self._model.parameters()
                                                    if param.get_device() != -1])[0]
            self._model.to('cpu'),
            logging.info(f"Model was allocated on CUDA id #{self._model_init_device_id}, moved to CPU.")

    def __restore_device(self):
        """
        Restoring the location of the model (on the CUDA) after its temporary transfer to the CPU.
        """
        if self._model_on_cuda and (self._model_init_device_id is not None):
            self._model.cuda(device=self._model_init_device_id)
            logging.info(f"Model transferred to CUDA id #{self._model_init_device_id}.")

    @staticmethod
    def __check_device_available(device_id: int = 0):
        """
        Returns a bool indicating if CUDA with provided id is currently available.
        """
        if torch.cuda.is_available():
            logging.info(f"CUDA device is available.")
            if device_id in list(range(torch.cuda.device_count())):
                try:
                    torch.cuda.set_device(device_id)
                except RuntimeError as e:
                    logging.error(f"CUDA error: cannot set device with provided id.")
                    return False
                return True
            else:
                logging.error(f"CUDA error: invalid device id.")
        logging.info(f"CUDA device #{device_id} is not available. Check configuration and try again.")
        return False

    def _get_parameter_sizes(self) -> List[Parameter]:
        """
        Get sizes of all parameters in `model`.
        Note: estimates only those layers that are included in model.modules().
        For example, use of nn.Functional.max_pool2d in the forward() method of a model prevents
        SizeEstimator from functioning properly.
        There is no direct means to access dimensionality changes
        carried out by arbitrary functions in the forward() method,
        such that tracking the size of inputs and gradients to be stored is non-trivial for such models.
        """
        sizes = []
        modules = list(self._model.modules())[1:]
        for i, module in enumerate(modules):
            # todo: handle torch.nn.ModuleDict!
            if isinstance(module, nn.ModuleList):
                # To not to estimate inner sub-modules twice!
                continue
            else:
                sizes.extend([Parameter(size=np.asarray(param.size()),
                                        bits=self.__get_parameter_bits(param))
                              for param in module.parameters()])
        return sizes

    def _get_output_sizes(self) -> List[Parameter]:
        """
        Run sample input through each layer to get output sizes
        """
        input_ = torch.FloatTensor(*self._input_size).requires_grad_(False)
        modules = list(self._model.modules())[1:]
        out_sizes = []
        for i, module in enumerate(modules):
            # todo: handle torch.nn.ModuleDict!
            if isinstance(module, nn.ModuleList):
                continue
            else:
                out = module(input_)
                out_sizes.append(Parameter(size=np.asarray(out.size()),
                                           bits=self.__get_parameter_bits(out)))
                input_ = out
        return out_sizes

    def _calculate_parameters_weight(self) -> float:
        """
        Calculate total number of bits to store `model` parameters
        """
        total_bits = 0
        for param in self._parameters_sizes:
            total_bits += np.prod(param.size) * param.bits
        return total_bits

    @staticmethod
    def __get_parameter_bits(param: torch.Tensor) -> int:
        """
        Calculate total number of bits to store `model` parameters
        """
        # Choose dtype
        if param.dtype == torch.float16:
            return 16
        elif param.dtype == torch.bfloat16:
            return 16
        elif param.dtype == torch.float32:
            return 32
        elif param.dtype == torch.float64:
            return 64
        else:
            logging.error(f"Current version estimated only sizes of floating points parameters!")
            return 32

    def _calculate_forward_backward_weight(self) -> float:
        """
        Calculate bits to store forward and backward pass
        """
        total_bits = []
        for out in self._output_sizes:
            # forward pass
            f_bits = np.prod(out.size) * out.bits
            total_bits.append(f_bits)

        # Multiply by 2 for both forward and backward
        return np.asarray(total_bits, dtype=np.float64).sum() * 2  # Cause overflow: total_bits * 2

    def _calculate_input_weight(self) -> float:
        """
        Calculate bits to store single input sequence.
        """
        return np.prod(np.array(self._input_size)) * self._input_n_bits

    def estimate_total_size(self) -> float:
        """
        Estimate total model size in memory in megabytes and bits.
        There are three main components which total size need to be estimated in memory during model training:
            * Model parameters: the actual weights in network;
            * Input: the input itself has to be estimated too (in case to not overflow GPU memory for example);
            * Intermediate variables: intermediate variables passed between layers, both the values and gradients;
        Therefore, this method returns total size of network (in Mb.) as sum of those three components.
        """
        total = self._input_weight + self._parameters_bits + self._forward_backward_bits
        total_bytes = (total / 8)
        total_megabytes = total_bytes / (1024**2)
        logging.info(f"Model size is: {total} bits, {total_bytes} bytes, {total_megabytes} Mb.")
        return total_megabytes

    def estimate_weights_size(self) -> float:
        """
        Estimate ONLY model weights size in memory in megabytes and bits.
        """
        total = self._input_weight
        total_bytes = (total / 8)
        total_megabytes = total_bytes / (1024 ** 2)
        logging.info(f"Model size is: {total} bits, {total_bytes} bytes, {total_megabytes} Mb.")
        return total_megabytes

    def estimate_cuda_memory(self, device_id: int, optimizer_type: torch.optim.Optimizer,
                             mixed_precision: bool = False) -> float:
        """
        Predict the maximum memory usage of the model,
        which is placed on CUDA device.
        GPU usage consist of three main components:
            * Model parameters: the actual weights in network;
            * Input: the main parameter of it is a batch size;
            * Intermediate variables:
                * for forward pass: activations of each weight;
                * for backward pass: gradients for each weight;
            * Optimizer: estimations of the first/second moments
                        of the gradient, for each model weight.
        Args:
            optimizer_type (Type): the class name of the optimizer to instantiate;
            mixed_precision (bool): whether to estimate based on using mixed precision;
            device_id (int): the device to use.
        """
        # Reset model and optimizer
        self.__restore_device()
        available_flg = SizeEstimator.__check_device_available(device_id)
        if not available_flg:
            logging.error(f"Cannot estimate memory usage of the model on CUDA, because it is not available.")
            return 0.0
        device = torch.cuda.device(f'cuda:{device_id}')
        # Releases all unoccupied cached memory currently held by the caching allocator
        torch.cuda.empty_cache()
        optimizer = optimizer_type(self._model.parameters(), lr=.001)

        # 1. Model weights size
        mem_0 = torch.cuda.memory_allocated(device_id)
        self._model.to(f'cuda:{device_id}')
        mem_1 = torch.cuda.memory_allocated(device)
        model_memory = mem_1 - mem_0

        # 2. Forward pass
        model_input = torch.FloatTensor(*self._input_size)
        output = self._model(model_input.to(f'cuda:{device_id}'))
        mem_2 = torch.cuda.memory_allocated(device)

        # 3. Backward pass
        if mixed_precision:
            mem_multiplier = .5
        else:
            mem_multiplier = 1
        forward_pass_memory = (mem_2 - mem_1) * mem_multiplier
        gradient_memory = model_memory

        # 4. Optimizer memory usage
        if isinstance(optimizer, torch.optim.Adam):
            o = 2
        elif isinstance(optimizer, torch.optim.RMSprop):
            o = 1
        elif isinstance(optimizer, torch.optim.SGD):
            o = 0
        else:
            raise ValueError(f"""Unsupported optimizer type. Look up how many moments are 
                             stored by your optimizer and add a case to the optimizer checker.""")
        gradient_moment_memory = o * gradient_memory

        # Total
        total = model_memory + forward_pass_memory + gradient_memory + gradient_moment_memory
        total_bytes = (total / 8)
        total_megabytes = total_bytes / (1024 ** 2)
        logging.info(f"Model size is: {total} bits, {total_bytes} bytes, {total_megabytes} Mb.")
        return total_megabytes

