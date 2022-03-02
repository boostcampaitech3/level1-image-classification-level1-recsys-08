import torch
from torchsampler import ImbalancedDatasetSampler

# https://dreamkkt.notion.site/ImbalancedDatasetSampler-094aca5628d246f39b1c72b9f9434014
class ImbalancedSampler(ImbalancedDatasetSampler):
    def _get_labels(self, dataset):
        if isinstance(dataset, torch.utils.data.Subset):
            return dataset.dataset.classes[dataset.indices]
        elif isinstance(dataset, torch.utils.data.Dataset):
            return dataset.classes
        else:
            raise NotImplementedError
