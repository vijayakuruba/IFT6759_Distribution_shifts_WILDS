import os, sys
import argparse
import wilds

def main():
    """
    Downloads the latest versions of all specified datasets,
    if they do not already exist.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--root_dir', required=True,
                        help='The directory where [dataset]/data can be found (or should be downloaded to, if it does not exist).')
    #parser.add_argument('--datasets', nargs='*', default="iwildcam",
    #                    help=f'Specify a space-separated list of dataset names to download. If left unspecified, the script will download all of the official benchmark datasets. Available choices are {wilds.supported_datasets}.')
    #parser.add_argument('--unlabeled', default=False, type=bool,
    #                    help=f'If this flag is set, the unlabeled dataset will be downloaded instead of the labeled.')
    #parser.add_argument('--datasets', default="iwildcam")
    config = parser.parse_args()

    print(f'Downloading the following dataset: iwildcam') 
    wilds.get_dataset(
            dataset="iwildcam",
            root_dir=config.root_dir,
            #unlabeled=config.unlabeled,
            download=True)


if __name__=='__main__':
    main()
