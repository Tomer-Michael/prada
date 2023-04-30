# todo tomer go over this file
from typing import Dict, List, Optional, Sequence, Tuple

import torch
from tqdm.auto import tqdm

from src.features_extraction.features_transforms import FeaturesTransform
from src.features_extraction.models import get_model_and_tokenizer, Model, SupportedModels, Tokenizer
from src.protein_datasets import Protein


class FeaturesExtractor:
    """
    Uses the tokenizer and the model to extract features for a single protein or a batch of proteins.
    Also applies a FeaturesTransform on the extracted features of each protein (separately).
    """
    def __init__(self, model: Model, tokenizer: Tokenizer, layers: Tuple[int, ...],
                 device: torch.device, features_transform: FeaturesTransform) -> None:
        self._model: Model = model
        self._tokenizer: Tokenizer = tokenizer
        self._layers: Tuple[int, ...] = tuple(sorted(layers))
        self._device: torch.device = device
        self._features_transform: FeaturesTransform = features_transform

    def extract_batch(self, proteins: Sequence[Protein], treat_errors_silently: bool = False) -> List[torch.Tensor]:
        """
        Extracts the features of all the proteins, a list of tensors of shape (num_channels, num_samples).
        """
        all_features = list()
        for protein in tqdm(proteins, desc="Getting features from model"):
            protein_features = self.extract(protein, treat_errors_silently)
            if protein_features is None:
                continue
            all_features.append(protein_features)
        return all_features

    def extract(self, protein: Protein, treat_errors_silently: bool = False) -> Optional[torch.Tensor]:
        """
        Extracts the protein's features, a tensor of shape (num_channels, num_samples).
        If treat_errors_silently is True and an error occurs, the faulty protein id will be logged and the function
        will return None.
        """
        try:
            protein_token = self._tokenizer(protein.id, protein.sequence)
            protein_token = protein_token.to(self._device)
            layer_to_features = self._model(protein_token, protein.sequence_len(), repr_layers=self._layers)
            cat_features = self._cat_layers(layer_to_features)
            features = self._features_transform(cat_features)
            return features
        except RuntimeError:
            print(f"Error while working on protein id [{protein.id}], protein sequence [{protein.sequence}].")
            if treat_errors_silently:
                return None
            else:
                raise

    def _cat_layers(self, layer_to_features: Dict[int, torch.Tensor]) -> torch.Tensor:
        all_features = list()
        for layer in self._layers:
            cur_features = layer_to_features[layer]
            all_features.append(cur_features)
        # Concatenate the channels from all the layers.
        features = torch.cat(all_features, dim=0)
        return features


def create_features_extractor(model_to_use: SupportedModels, layers: Tuple[int, ...],
                              device: torch.device, features_transform: FeaturesTransform) -> FeaturesExtractor:
    # todo tomer move elsewhere
    assert all(map(lambda layer: 0 <= layer <= model_to_use.number_of_layers, layers)), \
        f"Model supports layers in range [0, {model_to_use.number_of_layers}], but the requested layers are {layers}."

    model, tokenizer = get_model_and_tokenizer(model_to_use)
    print(f"Moving model to device {device}")
    model = model.to(device)

    return FeaturesExtractor(model, tokenizer, layers, device, features_transform)
