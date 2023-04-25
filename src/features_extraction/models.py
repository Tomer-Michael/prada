# todo tomer go over this file
from typing import Dict, Tuple, Union

import esm
import torch

from src.utils.enum_utils import EnumWithAttrs

_ESM_MODEL = Union[esm.model.ProteinBertModel, esm.model.MSATransformer]
_ESM_TOKENIZER = esm.Alphabet


class SupportedModels(EnumWithAttrs):
    def __init__(self,
                 full_model_name: str,
                 number_of_layers: int,
                 max_sequence_len: int,
                 embed_dim: int) -> None:
        self.full_model_name = full_model_name
        self.number_of_layers: int = number_of_layers
        self.max_sequence_len = max_sequence_len
        self.embed_dim = embed_dim  # Same for all layers in the model.

    # ESM Models. See https://github.com/facebookresearch/esm#main-models-you-should-use-
    ESM_1b = "esm1b_t33_650M_UR50S", 33, 1022, 1280


class Tokenizer:
    def __call__(self, protein_label: str, protein_sequence: str) -> torch.Tensor:
        pass


class Model(torch.nn.Module):
    def __call__(self, protein_token: torch.Tensor, protein_sequence_length: int,
                 repr_layers: Tuple[int, ...]) -> Dict[int, torch.Tensor]:
        pass


class _ESM1bTokenizerWrapper(Tokenizer):
    def __init__(self, backbone_tokenizer: _ESM_TOKENIZER) -> None:
        batch_converter = backbone_tokenizer.get_batch_converter()
        self._backbone_tokenizer: esm.BatchConverter = batch_converter

    def __call__(self, protein_label: str, protein_sequence: str) -> torch.Tensor:
        labels_and_sequences = [(protein_label, protein_sequence)]
        batched_labels, batched_strs, batched_tokens = self._backbone_tokenizer(labels_and_sequences)
        return batched_tokens


class _ESM1bModelWrapper(Model):
    _REPRESENTATIONS_KEY: str = "representations"

    def __init__(self, backbone_model: _ESM_MODEL) -> None:
        super().__init__()
        evaluated_model = backbone_model.eval()
        self._backbone_model = evaluated_model

    def __call__(self, protein_token: torch.Tensor, protein_sequence_length: int,
                 repr_layers: Tuple[int, ...]) -> Dict[int, torch.Tensor]:
        model_results = self._backbone_model(protein_token, repr_layers=repr_layers, return_contacts=False)
        representations = model_results[self._REPRESENTATIONS_KEY]
        features = dict()
        for layer in repr_layers:
            layer_features = representations[layer][0, 1: protein_sequence_length + 1]
            # Model outputs (batch_size, num_samples, num_channels), where batch_size is 1.
            # We want (num_channels, num_samples).
            layer_features = layer_features.squeeze().permute(1, 0)
            features[layer] = layer_features
        return features


def get_model_and_tokenizer(model_to_use: SupportedModels) -> Tuple[Model, Tokenizer]:
    print(f"Loading model and tokenizer for {model_to_use.name}")
    if model_to_use is SupportedModels.ESM_1b:
        backbone_model, backbone_tokenizer = _load_esm_model_and_tokenizer(model_to_use.full_model_name)
        model = _ESM1bModelWrapper(backbone_model)
        tokenizer = _ESM1bTokenizerWrapper(backbone_tokenizer)
    else:
        raise ValueError(f"Unsupported model: {model_to_use.name}")
    return model, tokenizer


def _load_esm_model_and_tokenizer(model_name: str) -> Tuple[_ESM_MODEL, _ESM_TOKENIZER]:
    return torch.hub.load(repo_or_dir="facebookresearch/esm:v0.4.2", model=model_name, source="github")