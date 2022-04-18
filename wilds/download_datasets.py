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
    config = parser.parse_args()

    print(f'Downloading the following dataset: iwildcam') 
    wilds.get_dataset(
            dataset="iwildcam",
            root_dir=config.root_dir,
            download=True)


if __name__=='__main__':
    main()
