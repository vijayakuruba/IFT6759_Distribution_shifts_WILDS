from typing import Optional

import wilds

def get_dataset(dataset: str, version: Optional[str] = None, **dataset_kwargs):
    """
    Returns the appropriate WILDS dataset class.
    Input:
        dataset (str): Name of the dataset
        version (Union[str, None]): Dataset version number, e.g., '1.0'.
                                    Defaults to the latest version.
        unlabeled (bool): If true, use the unlabeled version of the dataset.
        dataset_kwargs: Other keyword arguments to pass to the dataset constructors.
    Output:
        The specified WILDSDataset class.
    """
    if version is not None:
        version = str(version)

    if dataset not in wilds.supported_datasets:
        raise ValueError(f'The dataset {dataset} is not recognized. Must be one of {wilds.supported_datasets}.')

    elif dataset == 'iwildcam':
            if version == '1.0':
                from wilds.datasets.archive.iwildcam_v1_0_dataset import IWildCamDataset
            else:
                from wilds.datasets.iwildcam_dataset import IWildCamDataset # type:ignore
            return IWildCamDataset(version=version, **dataset_kwargs)

 
