import datetime
import torch
from utils.parser import get_parser_with_args
from utils.helpers import get_loaders, get_criterion, load_model
import os
import logging
import json
import random
import numpy as np
try:
    from contextlib import nullcontext
except ImportError:
    class nullcontext(object):
        def __enter__(self):
            return None

        def __exit__(self, *excinfo):
            return False

try:
    from tensorboardX import SummaryWriter
except ImportError:
    class SummaryWriter(object):
        def __init__(self, *args, **kwargs):
            logging.warning('tensorboardX is not installed; TensorBoard logging is disabled.')

        def add_scalars(self, *args, **kwargs):
            pass

        def close(self):
            pass

try:
    from tqdm import tqdm
except ImportError:
    class tqdm(object):
        def __init__(self, iterable):
            self.iterable = iterable

        def __iter__(self):
            return iter(self.iterable)

        def set_description(self, *args, **kwargs):
            pass


"""
Initialize Parser and define arguments
"""
parser, metadata = get_parser_with_args()
opt = parser.parse_args()

"""
Initialize experiments log
"""
logging.basicConfig(level=logging.INFO)
writer = SummaryWriter(opt.log_dir + f'/{datetime.datetime.now().strftime("%Y%m%d-%H%M%S")}/')

"""
Set up environment: define paths, download data, and set device
"""
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
dev = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
logging.info('GPU AVAILABLE? ' + str(torch.cuda.is_available()))
use_amp = bool(opt.amp and dev.type == 'cuda')
scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
amp_context = torch.cuda.amp.autocast if use_amp else nullcontext
logging.info('AMP ENABLED? ' + str(use_amp))

def seed_torch(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

seed_torch(seed=777)


train_loader, val_loader = get_loaders(opt)

"""
Load Model then define other aspects of the model
"""
logging.info('LOADING Model')
model = load_model(opt, dev)

criterion = get_criterion(opt)
optimizer = torch.optim.AdamW(model.parameters(), lr=opt.learning_rate) # Be careful when you adjust learning rate, you can refer to the linear scaling rule
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=8, gamma=0.5)

"""
 Set starting values
"""
best_metrics = {'cd_f1scores': -1, 'cd_recalls': -1, 'cd_precisions': -1}
logging.info('STARTING training')
total_step = -1
start_epoch = 0
checkpoint_dir = opt.checkpoint_dir
os.makedirs(checkpoint_dir, exist_ok=True)

if opt.resume:
    logging.info('LOADING checkpoint from ' + opt.resume)
    checkpoint = torch.load(opt.resume, map_location=dev)
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        if 'scaler_state_dict' in checkpoint:
            scaler.load_state_dict(checkpoint['scaler_state_dict'])
        best_metrics = checkpoint.get('best_metrics', best_metrics)
        total_step = checkpoint.get('total_step', total_step)
        start_epoch = checkpoint.get('epoch', -1) + 1
        logging.info('RESUMING from epoch ' + str(start_epoch))
    else:
        model = checkpoint.to(dev)
        logging.warning('Loaded legacy model checkpoint. Optimizer and scheduler states were not restored.')

def save_training_checkpoint(path, epoch, mean_val_metrics):
    checkpoint = {
        'epoch': epoch,
        'total_step': total_step,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'scaler_state_dict': scaler.state_dict(),
        'best_metrics': best_metrics,
        'validation_metrics': mean_val_metrics,
        'metadata': metadata,
    }
    torch.save(checkpoint, path)

def initialize_epoch_stats(device):
    return {
        'loss_sum': torch.zeros((), device=device),
        'corrects': torch.zeros((), device=device),
        'total': torch.zeros((), device=device),
        'tp': torch.zeros((), device=device),
        'fp': torch.zeros((), device=device),
        'fn': torch.zeros((), device=device),
        'tn': torch.zeros((), device=device),
        'num_batches': 0,
    }

def update_epoch_stats(stats, cd_loss, cd_preds, labels):
    labels = labels.squeeze(1) if labels.dim() == 4 and labels.size(1) == 1 else labels
    preds = cd_preds.squeeze(1) if cd_preds.dim() == 4 and cd_preds.size(1) == 1 else cd_preds
    preds = preds.long()
    labels = labels.long()

    stats['loss_sum'] += cd_loss.detach()
    stats['corrects'] += (preds == labels).sum()
    stats['total'] += labels.numel()
    stats['tp'] += ((preds == 1) & (labels == 1)).sum()
    stats['fp'] += ((preds == 1) & (labels == 0)).sum()
    stats['fn'] += ((preds == 0) & (labels == 1)).sum()
    stats['tn'] += ((preds == 0) & (labels == 0)).sum()
    stats['num_batches'] += 1

def finalize_epoch_stats(stats, lr):
    eps = 1e-7
    num_batches = max(stats['num_batches'], 1)
    loss = (stats['loss_sum'] / num_batches).item()
    corrects = (100.0 * stats['corrects'] / (stats['total'] + eps)).item()
    precision = (stats['tp'] / (stats['tp'] + stats['fp'] + eps)).item()
    recall = (stats['tp'] / (stats['tp'] + stats['fn'] + eps)).item()
    f1score = (2.0 * precision * recall / (precision + recall + eps))

    return {
        'cd_losses': loss,
        'cd_corrects': corrects,
        'cd_precisions': precision,
        'cd_recalls': recall,
        'cd_f1scores': f1score,
        'learning_rate': lr,
    }

for epoch in range(start_epoch, opt.epochs):
    train_metrics = initialize_epoch_stats(dev)
    val_metrics = initialize_epoch_stats(dev)

    """
    Begin Training
    """
    model.train()
    logging.info('SET model mode to train!')
    batch_iter = 0
    tbar = tqdm(train_loader)
    for batch_img1, batch_img2, labels in tbar:
        tbar.set_description("epoch {} info ".format(epoch) + str(batch_iter) + " - " + str(batch_iter+opt.batch_size))
        batch_iter = batch_iter+opt.batch_size
        total_step += 1
        # Set variables for training
        batch_img1 = batch_img1.float().to(dev)
        batch_img2 = batch_img2.float().to(dev)
        labels = labels.long().to(dev)

        # Zero the gradient
        optimizer.zero_grad()

        # Get model predictions, calculate loss, backprop
        with amp_context():
            cd_preds = model(batch_img1, batch_img2)
            cd_loss = criterion(cd_preds, labels)
        loss = cd_loss
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        cd_preds = cd_preds[-1]
        _, cd_preds = torch.max(cd_preds, 1)

        update_epoch_stats(train_metrics, cd_loss, cd_preds, labels)

        # clear batch variables from memory
        del batch_img1, batch_img2, labels

    scheduler.step()
    mean_train_metrics = finalize_epoch_stats(train_metrics, scheduler.get_last_lr()[0])
    logging.info("EPOCH {} TRAIN METRICS".format(epoch) + str(mean_train_metrics))

    for k, v in mean_train_metrics.items():
        writer.add_scalars(str(k), {'train': v}, total_step)

    """
    Begin Validation
    """
    model.eval()
    with torch.no_grad():
        for batch_img1, batch_img2, labels in val_loader:
            # Set variables for training
            batch_img1 = batch_img1.float().to(dev)
            batch_img2 = batch_img2.float().to(dev)
            labels = labels.long().to(dev)

            # Get predictions and calculate loss
            with amp_context():
                cd_preds = model(batch_img1, batch_img2)
                cd_loss = criterion(cd_preds, labels)

            cd_preds = cd_preds[-1]
            _, cd_preds = torch.max(cd_preds, 1)

            update_epoch_stats(val_metrics, cd_loss, cd_preds, labels)

            # clear batch variables from memory
            del batch_img1, batch_img2, labels

        mean_val_metrics = finalize_epoch_stats(val_metrics, scheduler.get_last_lr()[0])
        logging.info("EPOCH {} VALIDATION METRICS".format(epoch)+str(mean_val_metrics))

        for k, v in mean_val_metrics.items():
            writer.add_scalars(str(k), {'val': v}, total_step)

        metadata['validation_metrics'] = mean_val_metrics
        last_checkpoint_path = os.path.join(checkpoint_dir, 'last_checkpoint.pt')

        """
        Store the weights of good epochs based on validation results
        """
        if ((mean_val_metrics['cd_precisions'] > best_metrics['cd_precisions'])
                or
                (mean_val_metrics['cd_recalls'] > best_metrics['cd_recalls'])
                or
                (mean_val_metrics['cd_f1scores'] > best_metrics['cd_f1scores'])):

            # Insert training and epoch information to metadata dictionary
            logging.info('updata the model')

            # Save model and log
            with open(os.path.join(checkpoint_dir, 'metadata_epoch_' + str(epoch) + '.json'), 'w') as fout:
                json.dump(metadata, fout)

            best_metrics = mean_val_metrics
            save_training_checkpoint(os.path.join(checkpoint_dir, 'checkpoint_epoch_' + str(epoch) + '.pt'), epoch, mean_val_metrics)
            save_training_checkpoint(os.path.join(checkpoint_dir, 'best_checkpoint.pt'), epoch, mean_val_metrics)

            # comet.log_asset(upload_metadata_file_path)

        save_training_checkpoint(last_checkpoint_path, epoch, mean_val_metrics)


        print('An epoch finished.')
writer.close()  # close tensor board
print('Done!')
