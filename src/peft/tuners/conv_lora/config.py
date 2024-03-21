# Copyright 2023-present the HuggingFace Inc. team.
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

from dataclasses import dataclass, field
from typing import List, Optional, Union

from peft.config import PeftConfig
from peft.utils import PeftType
from peft.tuners.lora import LoraConfig

@dataclass
class ConvLoraConfig(LoraConfig):
    """
    This is the configuration class to store the configuration of a [`ConvLoraModel`].
    Args:
       
    """
    conv_lora_expert_num: int = field(default=8,
                                      metadata={"help": "The number of experts in the moe layer."}
                                )
    
