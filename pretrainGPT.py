import os
import numpy as np
import pickle
import platform
import multiprocessing

from mySettings import get_lstm_settings
from myTransformers import PretrainedGPT2
from myDataGenerator import lstmDataGenerator
from utilities import getAllMarkers, rotateArray
from torch import nn
import math

# %% User inputs.
# Select case you want to train, see mySettings for case-specific settings.
case = 0

runTraining = True
saveTrainedModel = True

# %% Paths.
if platform.system() == 'Linux':
    # To use docker.
   # pathMain = '/augmenter-cs230'   #comment by AKshit
    pathMain = os.getcwd() #Added by Akshit
else:
    pathMain = os.getcwd()
pathData = os.path.join(pathMain, "Data")
pathData_all = os.path.join(pathData, "data_CS230")
pathTrainedModels = os.path.join(pathMain, "trained_models_LSTM")
os.makedirs(pathTrainedModels, exist_ok=True)
pathCModel = os.path.join(pathTrainedModels, "")


settings = get_lstm_settings(case)
augmenter_type = settings["augmenter_type"]
poseDetector = settings["poseDetector"]
idxDatasets = settings["idxDatasets"]
scaleFactors = settings["scaleFactors"]
nHUnits = settings["nHUnits"]
nHLayers = settings["nHLayers"]
nEpochs = settings["nEpochs"]
#batchSize = settings["batchSize"]
idxFold = settings["idxFold"] 
mean_subtraction = settings["mean_subtraction"]
std_normalization = settings["std_normalization"]

# Learning rate.
learning_r = 1e-3 # default
if "learning_r" in settings:
    learning_r = settings["learning_r"] 
    
# Loss function.
loss_f = "mean_squared_error" # default
if "loss_f" in settings:
    loss_f = settings["loss_f"]
    
# Noise.
noise_bool = False # default
noise_magnitude = 0 # default
noise_type = '' # default
if "noise_magnitude" in settings:
    noise_magnitude = settings["noise_magnitude"]
    if noise_magnitude > 0:
        noise_bool = True
    noise_type = 'per_timestep'
    if 'noise_type' in settings:
        noise_type = settings["noise_type"]
        
# Rotation.
nRotations = 1
if "nRotations" in settings:
    nRotations = settings["nRotations"]
rotations = [i*360/nRotations for i in range(0,nRotations)]

# Bidirectional LSTM. 
# https://keras.io/api/layers/recurrent_layers/bidirectional/
bidirectional = False
if 'bidirectional' in settings:
    bidirectional = settings["bidirectional"]   
        
# %% Fixed settings (no need to change that).
featureHeight = True
featureWeight = True
marker_dimensions = ["x", "y", "z"]
nDim = len(marker_dimensions)        
fc = 60 # sampling frequency (Hz)
desired_duration = 0.5 # (s)
desired_nFrames = int(desired_duration * fc)

# Use multiprocessing (only working on Linux apparently).
# https://keras.io/api/models/model_training_apis/
if platform.system() == 'Linux':
    use_multiprocessing = True
    # Use all but 2 thread
    nWorkers = multiprocessing.cpu_count() - 2
else:
    # Not supported on Windows
    use_multiprocessing = False
    nWorkers = 1
    
# %% Helper indices (no need to change that).
# Get indices features/responses based on augmenter_type and poseDetector.
feature_markers_all, response_markers_all = getAllMarkers()
if augmenter_type == 'fullBody':
    if poseDetector == 'OpenPose':
        from utilities import getOpenPoseMarkers_fullBody
        _, _, idx_in_all_feature_markers, idx_in_all_response_markers = (
            getOpenPoseMarkers_fullBody())
    elif poseDetector == 'mmpose':
        from utilities import getMMposeMarkers_fullBody
        _, _, idx_in_all_feature_markers, idx_in_all_response_markers = (
            getMMposeMarkers_fullBody())
    else:
        raise ValueError('poseDetector not recognized')        
elif augmenter_type == 'lowerExtremity':
    if poseDetector == 'OpenPose':
        from utilities import getOpenPoseMarkers_lowerExtremity
        feature_markers, _, idx_in_all_feature_markers, idx_in_all_response_markers = (
            getOpenPoseMarkers_lowerExtremity())
    elif poseDetector == 'mmpose':
        from utilities import getMMposeMarkers_lowerExtremity
        _, _, idx_in_all_feature_markers, idx_in_all_response_markers = (
            getMMposeMarkers_lowerExtremity())
    else:
        raise ValueError('poseDetector not recognized')
elif augmenter_type == 'upperExtremity_pelvis':
    from utilities import getMarkers_upperExtremity_pelvis
    _, _, idx_in_all_feature_markers, idx_in_all_response_markers = (
        getMarkers_upperExtremity_pelvis())
elif augmenter_type == 'upperExtremity_noPelvis':
    from utilities import getMarkers_upperExtremity_noPelvis
    _, _, idx_in_all_feature_markers, idx_in_all_response_markers = (
        getMarkers_upperExtremity_noPelvis())
else:
    raise ValueError('augmenter_type not recognized')
    
# Each marker has 3 dimensions.
idx_in_all_features = []
for idx in idx_in_all_feature_markers:
    idx_in_all_features.append(idx*3)
    idx_in_all_features.append(idx*3+1)
    idx_in_all_features.append(idx*3+2)
idx_in_all_responses = []
for idx in idx_in_all_response_markers:
    idx_in_all_responses.append(idx*3)
    idx_in_all_responses.append(idx*3+1)
    idx_in_all_responses.append(idx*3+2)
    
# Additional features (height and weight). 
nAddFeatures = 0
nFeature_markers = len(idx_in_all_feature_markers)*nDim
nResponse_markers = len(idx_in_all_response_markers)*nDim
if featureHeight:
    nAddFeatures += 1
    idxFeatureHeight = len(feature_markers_all)*nDim
    idx_in_all_features.append(idxFeatureHeight)
if featureWeight:
    nAddFeatures += 1 
    if featureHeight:
        idxFeatureWeight = len(feature_markers_all)*nDim + 1
    else:
        idxFeatureWeight = len(feature_markers_all)*nDim
    idx_in_all_features.append(idxFeatureWeight)
        
# %% Partition (select data for training, validation, and test).
datasetName = ' '.join([str(elem) for elem in idxDatasets])
datasetName = datasetName.replace(' ', '_')
scaleFactorName = ' '.join([str(elem) for elem in scaleFactors])
scaleFactorName = scaleFactorName.replace(' ', '_')
scaleFactorName = scaleFactorName.replace('.', '')
partitionName = ('{}_{}_{}_{}'.format(augmenter_type, poseDetector, 
                                      datasetName, scaleFactorName))
pathPartition = os.path.join(pathData_all, 'partition_{}.npy'.format(partitionName))
if not os.path.exists(pathPartition):
    print('Computing partition')
    # Load splitting settings.
    subjectSplit = np.load(os.path.join(pathData, 'subjectSplit.npy'),
                           allow_pickle=True).item()    
    # Load infoData dict.
    infoData = np.load(os.path.join(pathData_all, 'infoData.npy'),
                       allow_pickle=True).item()
    # Get partition.
    from utilities import getPartition
    partition = getPartition(idxDatasets, scaleFactors, infoData,
                             subjectSplit, idxFold)    
    # Save partition.
    np.save(pathPartition, partition)
else:
    # Load partition.
    partition = np.load(pathPartition, allow_pickle=True).item()
    
# %% Data processing: add noise and compute mean and standard deviation.
pathMean = os.path.join(pathData_all, 'mean_{}_{}_{}_{}.npy'.format(
    partitionName, noise_type, noise_magnitude, nRotations))
pathSTD = os.path.join(pathData_all, 'std_{}_{}_{}_{}.npy'.format(
    partitionName, noise_type, noise_magnitude, nRotations))
if not os.path.exists(pathMean) and not os.path.exists(pathSTD):
    print('Computing mean and standard deviation')
    # Instead of accumulating data to compute mean and std, we compute them
    # on the fly so that we do not have to deal with too large matrices.
    existingAggregate = (0,0,0)    
    from utilities import update
    from utilities import finalize
    for count, idx in enumerate(partition['train']): 
        c_features_all = np.load(os.path.join(
            pathData_all, "feature_{}.npy".format(idx)))
        # Select features.
        c_features = c_features_all[:,idx_in_all_features]
        # Apply rotations.
        for rotation in rotations:            
            c_features_xyz_rot = rotateArray(
                c_features[:,:nFeature_markers], 'y', rotation)
            c_features_rot = np.concatenate(
                (c_features_xyz_rot, c_features[:,nFeature_markers:]), axis=1)        
            # Add noise to the training data.
            # Here we adjust the np.random.seed at each iteration such that we 
            # can use the exact same noise in the data generator.
            if noise_magnitude > 0:
                if noise_type == "per_timestep": 
                    np.random.seed(idx)
                    noise = np.zeros((desired_nFrames,
                                      nFeature_markers+nAddFeatures))
                    noise[:,:nFeature_markers] = np.random.normal(
                        0, noise_magnitude, 
                        (desired_nFrames, nFeature_markers))
                    c_features += noise                
                else:
                    raise ValueError("Only per_timestep noise type supported")                
            if not c_features.shape[0] == 30:
                raise ValueError("Dimension features and responses are wrong")                
            # Compute mean and std iteratively.    
            for c_s in range(c_features.shape[0]):
                existingAggregate = update(existingAggregate, c_features[c_s, :])
    # Compute final mean and standard deviation.
    (features_mean, features_variance, _) = finalize(existingAggregate)    
    features_std = np.sqrt(features_variance)
    np.save(pathMean, features_mean)
    np.save(pathSTD, features_std)
else:
    features_mean = np.load(pathMean)
    features_std = np.load(pathSTD)
    
# %% Initialize data generators.
# https://stanford.edu/~shervine/blog/keras-how-to-generate-data-on-the-fly


import matplotlib.pyplot as plt
import numpy as np
import torch
import gc

gc.collect()

torch.cuda.empty_cache()

if torch.cuda.is_available():
    device = 'cuda'
else:
    device = 'cpu'

new_features = [("Neck", "RShoulder"),
                ("Neck", "LShoulder"),
                ("RShoulder", "LShoulder"),
#                ("RShoulder", "RElbow"),
#                ("LShoulder", "LElbow"),
#                ("RElbow", "RWrist"),
#                ("LElbow", "LWrist"),
                ("RShoulder", "RHip"),
                ("LShoulder", "LHip"),
                ("LHip", "RHip"),
                ("LHip", "LKnee"),
                ("RHip", "RKnee"),
                ("RKnee", "RAnkle"),
                ("LKnee", "LAnkle"),
                ("RAnkle", "RHeel"),
                ("LAnkle", "LHeel"),
                ("RHeel", "RSmallToe"),
                ("LHeel", "LSmallToe"),
                ("RHeel", "RBigToe"),
                ("LHeel", "LBigToe")]


##THESE ARE THE ONLY HYPERPARAMETERS THAT MATTER NOW
intermediate_width = 256
activation = "sigmoid" # can be "relu", "sigmoid", or "tanh"
loss_function = "rmse" # can be "rmse", "l1", or "log_cosh"
batch_size = 128
learning_rate = 1e-3
use_novel = False
gpt_frozen = True
##END OF PARAMETERS


params = {'dim_f': (desired_nFrames,nFeature_markers+nAddFeatures),
          'dim_r': (desired_nFrames,nResponse_markers),
          'batch_size': batch_size,
          'shuffle': True,
          'noise_bool': noise_bool,
          'noise_type': noise_type,
          'noise_magnitude': noise_magnitude,
          'mean_subtraction': mean_subtraction,
          'std_normalization': std_normalization,
          'features_mean': features_mean,
          'features_std': features_std,
          'idx_features': idx_in_all_features,
          'idx_responses': idx_in_all_responses,
          'rotations': rotations}
train_generator = lstmDataGenerator(partition['train'], pathData_all, **params)
val_generator = lstmDataGenerator(partition['val'], pathData_all, **params)

src_dim = nFeature_markers + nAddFeatures
trg_dim = nResponse_markers
if use_novel:
    src_dim += len(new_features)

model = PretrainedGPT2(src_dim = src_dim, trg_dim = trg_dim, output_hidden_dim = intermediate_width, dropout = 0.1, activation = activation, gpt_frozen = gpt_frozen).to(device)

mse_loss = nn.MSELoss()
def rmse_loss(yhat,y):
    return torch.sqrt(mse_loss(yhat, y))

l1_loss = nn.L1Loss()

def log_cosh_loss(y_pred, y_true):
    def _log_cosh(x):
        return x + torch.nn.functional.softplus(-2. * x) - math.log(2.0)
    return torch.mean(_log_cosh(y_pred - y_true))

if loss_function == "rmse":
    loss_fc = rmse_loss
elif loss_function == "l1":
    loss_fc = l1_loss
elif loss_function == "log_cosh":
    loss_fc = log_cosh_loss
else:
    raise Exception(f"Unkown loss {loss_function}")

new_features_idx = [(feature_markers.index(x), feature_markers.index(y)) for x,y in new_features]
new_features_idx = [[x*3, x*3+1, x*3+2] for x,_ in new_features_idx], [[y*3, y*3+1, y*3+2] for _,y in new_features_idx]
new_features_idx = torch.tensor(new_features_idx[0]).to(device), torch.tensor(new_features_idx[1]).to(device)

def add_novelty(x, new_features_idx):
    novel_features = (x[:,:, new_features_idx[0]] - x[:,:,new_features_idx[1]]).norm(dim = 3)
    return torch.cat((x, novel_features), dim = 2)

optimizer = torch.optim.AdamW(model.parameters(), lr = learning_rate)

print(f"Batch size: {batch_size}")
print(f"Train steps per epoch: {len(train_generator)}")

for epoch in range(nEpochs):
    print(f"XXXXXXXXXXXXXXXXXXXXX Epoch {epoch+1}/{nEpochs}")
    running_loss = 0.0
    running_rmse = 0.0
    model.train()
    for i in range(len(train_generator)):
        optimizer.zero_grad()

        x,y = train_generator[i]

        x = torch.from_numpy(x).to(device=device, dtype=torch.float32)
        y = torch.from_numpy(y).to(device=device, dtype=torch.float32)

        if use_novel:
            x = add_novelty(x, new_features_idx)
    
        output = model(x)
    
        loss = loss_fc(output,y)
        rmse = rmse_loss(output,y)
    
        loss.backward()
        running_loss += float(loss)
        running_rmse += float(rmse)
        optimizer.step()
        
        if i % 100 == 99:
            print(f"epoch: {epoch+1}, step: {i+1}, rmse_loss: {float(rmse):.8f}, {loss_function}_loss {float(loss):.8f}")

    train_loss = running_loss  / len(train_generator)
    train_rmse = running_rmse / len(train_generator)
    
    running_loss = 0.0
    running_rmse = 0.0
    with torch.no_grad():
        model.eval()
        for i in range(len(val_generator)):
            x,y = val_generator[i]
    
            x  = torch.from_numpy(x).to(device=device, dtype=torch.float32)
            y  = torch.from_numpy(y).to(device=device, dtype=torch.float32)
    
            if use_novel:
                x = add_novelty(x, new_features_idx)

            output = model(x)
    
            loss = loss_fc(output,y)
            rmse = rmse_loss(output,y)
            running_loss += float(loss)
            running_rmse += float(rmse)

        eval_loss = running_loss  / len(val_generator)
        eval_rmse = running_rmse / len(val_generator)
        print(f"epoch: {epoch+1}, train rmse_loss: {float(train_rmse):.8f}, train {loss_function}_loss {float(train_loss):.8f}, eval rmse_loss: {float(eval_rmse):.8f}, eval {loss_function}_loss {float(eval_loss):.8f}")

    train_generator.on_epoch_end()
    val_generator.on_epoch_end()

