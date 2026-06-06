'''
This file is used to save the output image
'''

import torch
import torch.utils.data
from utils.parser import get_parser_with_args
from utils.helpers import get_test_loaders, initialize_metrics, load_model
import os
from tqdm import tqdm
import cv2

if not os.path.exists('./output_img'):
    os.mkdir('./output_img')

parser, metadata = get_parser_with_args()
opt = parser.parse_args()

dev = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

test_loader = get_test_loaders(opt, batch_size=1)

path = 'models\checkpoint_epoch_95.pt'   # the path of the model
# Load model properly from checkpoint
checkpoint = torch.load(path, map_location=dev)
model = load_model(opt, dev)

if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
    model.load_state_dict(checkpoint['model_state_dict'])
else:
    # Legacy checkpoint - directly use as model
    model = checkpoint.to(dev)

model.eval()
index_img = 0
test_metrics = initialize_metrics()
with torch.no_grad():
    tbar = tqdm(test_loader)
    for batch_img1, batch_img2, labels in tbar:

        batch_img1 = batch_img1.float().to(dev)
        batch_img2 = batch_img2.float().to(dev)
        labels = labels.long().to(dev)

        cd_preds = model(batch_img1, batch_img2)

        cd_preds = cd_preds[-1]
        _, cd_preds = torch.max(cd_preds, 1)
        cd_preds = cd_preds.data.cpu().numpy()
        cd_preds = cd_preds.squeeze() * 255

        file_path = './output_img/' + str(index_img).zfill(5)
        cv2.imwrite(file_path + '.png', cd_preds)

        index_img += 1