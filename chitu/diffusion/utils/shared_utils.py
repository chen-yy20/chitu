import torch
from typing import List, Dict, Tuple, Optional

class SequencePadder:
    _padding_info: Dict = {}
    
    @staticmethod
    def split_sequence_padding(tensor: torch.Tensor, 
                             split_dim: int = 0,
                             name: Optional[str] = None) -> List[torch.Tensor]:
        """Split tensor along specified dimension with padding if needed"""
        size = tensor.size(split_dim)
        split_size = (size + 1) // 2  # Split into roughly equal halves

        if size % split_size != 0:
            pad_size = split_size - (size % split_size)
            pad_shape = list(tensor.shape)
            pad_shape[split_dim] = pad_size
            padding = torch.zeros(pad_shape, dtype=tensor.dtype, device=tensor.device)
            tensor = torch.cat([tensor, padding], dim=split_dim)
            if name is not None:
                SequencePadder._padding_info[name] = {'original_size': size, 'pad_size': pad_size}
    
        splits = torch.split(tensor, split_size, dim=split_dim)
        return list(splits)

    @staticmethod
    def remove_sequence_padding_and_concat(tensor_list: List[torch.Tensor], 
                              name: str,
                              dim: int = 0) -> torch.Tensor:
        """Remove padding from split tensors based on padding info"""
        if name not in SequencePadder._padding_info:
            return torch.cat(tensor_list, dim=dim)
            
        info = SequencePadder._padding_info[name]
        original_size = info['original_size']
        
        tensor = torch.cat(tensor_list, dim=dim)
        slicing = [slice(None)] * tensor.dim()
        slicing[dim] = slice(0, original_size)
        tensor = tensor[slicing]
        
        return tensor