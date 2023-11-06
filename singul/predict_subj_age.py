from dp_model.model_files.sfcn import SFCN
from dp_model import dp_loss as dpl
from dp_model import dp_utils as dpu
from nilearn.image import get_data, load_img
import torch
import torch.nn.functional as F
import os.path as op
import numpy as np
import argparse
from pathlib import Path
import csv

parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--s_path', metavar='N', type=str,
                    help='an integer for the accumulator')

args = parser.parse_args()

print(Path(args.s_path))


model = SFCN()
model = torch.nn.DataParallel(model)
fp_ = '/dp/UKBiobank_deep_pretrain-master/brain_age/run_20190719_00_epoch_best_mae.p'
model.load_state_dict(torch.load(fp_))
model.cuda()

# load subject vol
subj_img = load_img(args.s_path)
subj_name = op.basename(args.s_path)
subj_name = op.basename(subj_name).split('.', 1)[0] 
data = get_data(subj_img)

# Transforming the age to soft label (probability distribution)
bin_range = [42,82]
bin_step = 1
sigma = 1
bin_start = bin_range[0]
bin_end = bin_range[1]
bin_length = bin_end - bin_start
bin_number = int(bin_length / bin_step)
bin_centers = bin_start + float(bin_step) / 2 + bin_step * np.arange(bin_number)

# Preprocessing
data = data/data.mean()
data = dpu.crop_center(data, (160, 192, 160))

# Move the data from numpy to torch tensor on GPU
sp = (1,1)+data.shape
data = data.reshape(sp)
input_data = torch.tensor(data, dtype=torch.float32).cuda()

# Evaluation
model.eval() # Don't forget this. BatchNorm will be affected if not in eval mode.
with torch.no_grad():
    output = model(input_data)

# Output, loss, visualisation
x = output[0].cpu().reshape([1, -1])

# Prediction, Visualisation and Summary
x = x.numpy().reshape(-1)
prob = np.exp(x)
pred = prob@bin_centers
out_file = '/tmp/' + subj_name + ".csv"
with open(out_file, 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['nii', 'predicted age'])
    writer.writerow([args.s_path, round(pred, 3)])