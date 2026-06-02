import argparse as ag
import json

def get_parser_with_args(metadata_json='metadata.json'):
    parser = ag.ArgumentParser(description='Training change detection network')

    with open(metadata_json, 'r') as fin:
        metadata = json.load(fin)
        parser.set_defaults(**metadata)

        parser.add_argument('--n_channels', type=int, dest='num_channel', help='number of channels')
        parser.add_argument('--batch_size', type=int, dest='batch_size', help='batch size')
        parser.add_argument('--crop_size', '--patch_size', type=int, dest='patch_size', help='crop/patch size')
        parser.add_argument('--epochs', type=int, dest='epochs', help='number of epochs')
        parser.add_argument('--lr', type=float, dest='learning_rate', help='learning rate')
        parser.add_argument('--dataset_dir', type=str, dest='dataset_dir', help='dataset root directory')
        parser.add_argument('--log_dir', type=str, dest='log_dir', help='tensorboard log directory')
        parser.add_argument('--amp', action='store_true', dest='amp', help='enable CUDA automatic mixed precision')
        parser.add_argument('--no-amp', action='store_false', dest='amp', help='disable CUDA automatic mixed precision')

        return parser, metadata

    return None
