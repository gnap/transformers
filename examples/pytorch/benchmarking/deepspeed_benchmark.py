#!/usr/bin/env python
# coding=utf-8
# Copyright 2020 The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" Benchmarking the library on inference and training """

import deepspeed
from dataclasses import dataclass, field
from typing import Tuple
from typing import Callable, Optional
from transformers.file_utils import cached_property, is_torch_available, is_torch_tpu_available, torch_required
from transformers.utils import logging
from transformers import HfArgumentParser, PyTorchBenchmark, PyTorchBenchmarkArguments, PretrainedConfig
from transformers.training_args import TrainingArguments
from transformers.benchmark.benchmark_args_utils import BenchmarkArguments
from transformers.models.auto.modeling_auto import MODEL_MAPPING, MODEL_WITH_LM_HEAD_MAPPING
import torch.optim as optim
import torch.distributed as dist

if is_torch_available():
    import torch

if is_torch_tpu_available():
    import torch_xla.core.xla_model as xm

logger = logging.get_logger(__name__)

@dataclass
class CustomBenchmarkArguments(PyTorchBenchmarkArguments):

    def __init__(self, **kwargs):
        self.local_rank = kwargs.pop("local_rank", self.local_rank)
        self.deepspeed_config = kwargs.pop("deepspeed_config", self.deepspeed_config)
        super().__init__(**kwargs)

    deepspeed_config: str = field(default='tests/deepspeed/ds_config_zero3.json', metadata={"help": "deepspeed_config file"})
    local_rank: int = field(default=0, metadata={"help": "local rank of the worker process"})

    @cached_property
    @torch_required
    def _setup_devices(self) -> Tuple["torch.device", int]:
        logger.info("PyTorch: setting up devices")
        if not self.cuda:
            device = torch.device("cpu")
            n_gpu = 0
        elif is_torch_tpu_available():
            device = xm.xla_device()
            n_gpu = 0
        else:
            # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            device = torch.device("cuda:" + str(self.local_rank) if torch.cuda.is_available() else "cpu")
            n_gpu = torch.cuda.device_count()
        return device, n_gpu

    @property
    @torch_required
    def device(self) -> "torch.device":
        return self._setup_devices[0]

    @property
    def device_idx(self) -> int:
        # TODO(PVP): currently only single GPU is supported
        # return torch.cuda.current_device()
        return self.local_rank

class CustomBenchmark(PyTorchBenchmark):
    training_args: TrainingArguments

    def __init__(self, args: BenchmarkArguments = None, configs: PretrainedConfig = None, training_args: TrainingArguments = None):
        self.training_args = training_args
        super(self, PyTorchBenchmark).__init__(args=args, configs=configs)

    def _train_speed(self, model_name: str, batch_size: int, sequence_length: int) -> float:
        _train = self._prepare_train_func(model_name, batch_size, sequence_length)
        speed = self._measure_speed(_train)
        return speed

    def _prepare_train_func(self, model_name: str, batch_size: int, sequence_length: int) -> Callable[[], None]:
        config = self.config_dict[model_name]

        has_model_class_in_config = (
            hasattr(config, "architectures")
            and isinstance(config.architectures, list)
            and len(config.architectures) > 0
        )
        if not self.args.only_pretrain_model and has_model_class_in_config:
            try:
                model_class = config.architectures[0]
                transformers_module = __import__("transformers", fromlist=[model_class])
                model_cls = getattr(transformers_module, model_class)
                model = model_cls(config)
            except ImportError:
                raise ImportError(
                    f"{model_class} does not exist. If you just want to test the pretrained model, you might want to set `--only_pretrain_model` or `args.only_pretrain_model=True`."
                )
        else:
            model = MODEL_WITH_LM_HEAD_MAPPING[config.__class__](config)

        if self.args.torchscript:
            raise NotImplementedError("Training for torchscript is currently not implemented")
        else:
            train_model = model

        model.train()
        model.to(self.args.device)

        from transformers.deepspeed import HfTrainerDeepSpeedConfig

        # will be used later by the Trainer
        # note: leave self.deepspeed unmodified in case a user relies on it not to be modified)
        self.hf_deepspeed_config = HfTrainerDeepSpeedConfig('tests/deepspeed/ds_config_zero3.json')
        self.hf_deepspeed_config.trainer_config_process(self.training_args)

        model_engine, optimizer, _, _ = deepspeed.initialize(config_params=self.hf_deepspeed_config.config,
                                                             model=model,
                                                             model_parameters=model.parameters())

        # encoder-decoder has vocab size saved differently
        vocab_size = config.vocab_size if hasattr(config, "vocab_size") else config.encoder.vocab_size
        input_ids = torch.randint(vocab_size, (batch_size, sequence_length), dtype=torch.long, device=self.args.device)


        def compute_loss_and_backprob_encoder():
            loss = model_engine(input_ids, labels=input_ids)[0]
            loss.backward()
            optimizer.step()
            return loss

        def compute_loss_and_backprob_encoder_decoder():
            loss = model_engine(input_ids, decoder_input_ids=input_ids, labels=input_ids)[0]
            loss.backward()
            optimizer.step()
            return loss

        _train = (
            compute_loss_and_backprob_encoder_decoder
            if config.is_encoder_decoder
            else compute_loss_and_backprob_encoder
        )
        return _train

def main():
    parser = HfArgumentParser((CustomBenchmarkArguments, TrainingArguments))
    try:
        benchmark_args, training_args = parser.parse_args_into_dataclasses()[:1]
    except ValueError as e:
        arg_error_msg = "Arg --no_{0} is no longer used, please use --no-{0} instead."
        begin_error_msg = " ".join(str(e).split(" ")[:-1])
        full_error_msg = ""
        depreciated_args = eval(str(e).split(" ")[-1])
        wrong_args = []
        for arg in depreciated_args:
            # arg[2:] removes '--'
            if arg[2:] in PyTorchBenchmarkArguments.deprecated_args:
                # arg[5:] removes '--no_'
                full_error_msg += arg_error_msg.format(arg[5:])
            else:
                wrong_args.append(arg)
        if len(wrong_args) > 0:
            full_error_msg = full_error_msg + begin_error_msg + str(wrong_args)
        raise ValueError(full_error_msg)

    deepspeed.init_distributed()

    benchmark = CustomBenchmark(args=benchmark_args, training_args=training_args)
    benchmark.run()


if __name__ == "__main__":
    main()
