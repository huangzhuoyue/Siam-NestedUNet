import argparse as ag
import json

def get_parser_with_args(metadata_json='metadata.json'):
    parser = ag.ArgumentParser(description='Training change detection network')

    with open(metadata_json, 'r') as fin:
        metadata = json.load(fin)
        parser.set_defaults(**metadata)

        parser.add_argument('--n_channels', type=int, dest='num_channel', help='number of channels')
        parser.add_argument('--batch_size', type=int, dest='batch_size', help='batch size')
        parser.add_argument('--crop_size', type=int, dest='patch_size', help='crop size')
        parser.add_argument('--epochs', type=int, dest='epochs', help='number of epochs')
        parser.add_argument('--lr', type=float, dest='learning_rate', help='learning rate')

        return parser, metadata

    return None
