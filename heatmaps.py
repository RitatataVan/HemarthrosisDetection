''' File to run heatmapping extraction. '''
import pytorch_lightning as pl
from matplotlib import cm
import cv2
from PIL import Image
import numpy as np
import pandas as pd
import torch
from torchvision import transforms

from network import Net
from hyperparameters import parameters as params

class saveFeatures():
    '''
        Class to put a hook up on the model so we
        can get the features for the heatmaps.
    '''
    def __init__(self, m):
        self.hook = m.register_forward_hook(self.hook_fn)

    def hook_fn(self, module, input, output):
        self.features = ((output.cpu()).data).numpy()

    def remove(self):
        self.hook.remove()
        
def getCAM(feature_conv, weight_fc):
    '''
        Function to retrieve and calculate heatmaps.
        
        Args:
            - feature_conv: Features from the desidered convolutional layer
            - weight_fc: FC layer in charge of the final prediction
            - class_: Class to focus the heatmap on
                
        Return:
            - cam_img: A heatmap for a given image and a class
    '''
    _, nc, h, w = feature_conv.shape
    cam = weight_fc.dot(feature_conv.reshape((nc, h*w)))
    cam = cam.reshape(h, w)
    cam = cam - np.min(cam)
    cam_img = cam / np.max(cam)
    cam_img = cm.jet_r(cam_img)[..., :3] * 255
    cam_img *= 0.5
    return cam_img

def get_heatmaps(model, features, image, idx):
    # Heatmaps
    fc_params = list(model.model._modules.get('_fc').parameters())
    fc = np.squeeze(fc_params[0].cpu().data.numpy())
    heatmap = getCAM(features.features, fc)

    # Rescale output
    image_c = np.array(image)
    h,w,c = image_c.shape
    heatmap = cv2.resize(heatmap, (w,h))

    # Merge original and heatmap
    heatmap = (heatmap.astype(np.float) + image_c) / 2

    # Transform to bytes
    heatmap = cv2.cvtColor(heatmap.astype("uint8"), cv2.COLOR_BGR2RGB)
    heatmap = Image.fromarray(heatmap)
    heatmap.save(f'../heatmap_results/heatmap_{idx}.png')  
        
def to_tensor(image, **kwargs):
    """
       Function to change image channels to Pytorch format

       image: Image to be formatted
    """
    return image.transpose(2, 0, 1).astype('float32')


pl.seed_everything(params['seed'])

# Transforms
in_transform = transforms.Compose([
            transforms.Resize((params['img_size'], params['img_size'])),
            transforms.ToTensor(),
            transforms.Normalize(params['data_mean'], params['data_std'])])
out_transform = torch.nn.Sigmoid()

# setup model
model = Net.load_from_checkpoint('../weights_syn/adtblood-syn-epoch=42.ckpt').t$
model.eval()
model.freeze()

# Hook up for heatmaps
feature_layer = list(model.model.children())[-5]
features = saveFeatures(feature_layer)

# Predict Image
test_data = pd.read_csv('../datasets_syn/testing_data.csv')
test_data = test_data.sample(frac=1).reset_index(drop=True)
rows = []

for idx, row in test_data.iterrows():
    filename, label = row['ID_IMG'], row['BLOOD']
    print(filename)

    image = Image.open(filename).convert('RGB')
    img = in_transform(image)
    img = img.type(torch.float32).unsqueeze(0).to('cuda')
    output = model(img)
    probs = out_transform(output).cpu().item()
    prediction = 0 if probs < 0.5 else 1
    
    get_heatmaps(model, features, image, idx)
    
    rows.append({'ID_IMG':filename, 'BLOOD':label, 'PROB':probs, 'PRED':predict$
    
    if idx == 50:
        break
    
df = pd.DataFrame(rows, columns=['ID_IMG','BLOOD','PROB','PRED'])    
df.to_csv(f"../heatmap_results/test_results_heatmaps.csv")