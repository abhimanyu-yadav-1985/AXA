"""
This is an example using CLAP to perform zeroshot
    classification on ESC50 (https://github.com/karolpiczak/ESC-50).
"""

from msclap import CLAP
from dataset import ESC50
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from sklearn.metrics import accuracy_score

# Load dataset
root_path = "dataset" # Folder with ESC-50-master/
dataset = ESC50(root=root_path, download=True) #If download=False code assumes base_folder='ESC-50-master' in esc50_dataset.py
prompt = 'this is the sound of '
y = [prompt + x for x in dataset.classes]

clap_model = CLAP(version = '2023', use_cuda=False)

# Computing text embeddings
text_embeddings = clap_model.get_text_embeddings(y)

# Extract audio embeddings
audio_embeddings = clap_model.get_audio_embeddings([r"C:\PythonCode\AudioClassifier\dataset\ESC-50-master\audio\1-137-A-32.wav"])

# Compute similarity between audio and text embeddings 
similarity= clap_model.compute_similarity(audio_embeddings, text_embeddings)


    
y_pred = F.softmax(similarity.detach().cpu(), dim=1).numpy()
print(y[np.argmax(y_pred)])