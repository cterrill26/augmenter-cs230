import os
import numpy as np
import pickle
import platform
import multiprocessing
import tensorflow as tf

from mySettings import get_lstm_settings
from myModels import get_lstm_model
from myDataGenerator import lstmDataGenerator
from utilities import getAllMarkers, rotateArray

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
batchSize = settings["batchSize"]
print(batchSize)
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
        _, _, idx_in_all_feature_markers, idx_in_all_response_markers = (
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
params = {'dim_f': (desired_nFrames,nFeature_markers+nAddFeatures),
          'dim_r': (desired_nFrames,nResponse_markers),
          'batch_size': batchSize,
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


import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

from transformers.models.gpt2.modeling_gpt2 import GPT2Model


import gc

gc.collect()

torch.cuda.empty_cache()

if torch.cuda.is_available():
    device = 'cuda'
else:
    device = 'cpu'

gpt2 = GPT2Model.from_pretrained('gpt2')  # loads a pretrained GPT-2 base model
in_layer = nn.Linear(63, 768)           # map bit to GPT-2 embedding dim of 768
out_layer = nn.Linear(768, 63)# predict logits
#out_layer_relu = nn.functional.relu()
out_layer2 = nn.Linear(63, 105)
#out_layer2_relu = nn.functional.relu()
#out_layer3 = nn.Linear(80,105)



for name, param in gpt2.named_parameters():
    # freeze all parameters except the layernorm and positional embeddings
    if 'ln' in name or 'wpe' in name:
        param.requires_grad = True
    else:
        #param.requires_grad = False
        param.requires_grad = False


def RMSELoss(yhat,y):
    return torch.sqrt(torch.mean((yhat-y)**2))

def l1_Loss(yhat,y):
    return (torch.mean(torch.abs(yhat-y)))


params = list(gpt2.parameters()) + list(in_layer.parameters()) + list(out_layer.parameters()) + list(out_layer2.parameters())
optimizer = torch.optim.Adam(params)

for layer in (gpt2, in_layer, out_layer, out_layer2):
    layer.to(device=device)
    layer.train()

print(len(train_generator))

for epoch in range(nEpochs):
    print("XXXXXXXXXXXXXXXXXXXXX Epoch No. =  ", epoch)
    running_loss_plot = 0.0
    for i in range(len(train_generator)):
        data = train_generator[i]
        x = (data[0])
        y = (data[1])

        neck_rShoulder = np.sqrt((x[:,:,0,None] - x[:,:,3,None])**2 + (x[:,:,1,None] - x[:,:,4,None])**2 + (x[:,:,2,None] - x[:,:,5,None])**2)
        neck_lShoulder = np.sqrt((x[:,:,0,None] - x[:,:,6,None])**2 + (x[:,:,1,None] - x[:,:,7,None])**2 + (x[:,:,2,None] - x[:,:,8,None])**2)
        rShoulder_lShoulder = np.sqrt((x[:,:,3,None] - x[:,:,6,None])**2 + (x[:,:,4,None] - x[:,:,7,None])**2 + (x[:,:,5,None] - x[:,:,8,None])**2)
        rShoulder_rHip = np.sqrt((x[:,:,3,None] - x[:,:,9,None])**2 + (x[:,:,4,None] - x[:,:,10,None])**2 + (x[:,:,5,None] - x[:,:,11,None])**2)
        lShoulder_lHip = np.sqrt((x[:,:,6,None] - x[:,:,12,None])**2 + (x[:,:,7,None] - x[:,:,13,None])**2 + (x[:,:,8,None] - x[:,:,14,None])**2)
        rHip_lHip = np.sqrt((x[:,:,9,None] - x[:,:,12,None])**2 + (x[:,:,10,None] - x[:,:,13,None])**2 + (x[:,:,11,None] - x[:,:,14,None])**2)
        rHip_rKnee = np.sqrt((x[:,:,9,None] - x[:,:,15,None])**2 + (x[:,:,10,None] - x[:,:,16,None])**2 + (x[:,:,11,None] - x[:,:,17,None])**2)
        lHip_lKnee = np.sqrt((x[:,:,12,None] - x[:,:,18,None])**2 + (x[:,:,13,None] - x[:,:,19,None])**2 + (x[:,:,14,None] - x[:,:,20,None])**2)
        rKnee_rAnkle = np.sqrt((x[:,:,15,None] - x[:,:,21,None])**2 + (x[:,:,16,None] - x[:,:,22,None])**2 + (x[:,:,17,None] - x[:,:,23,None])**2)
        lKnee_lAnkle = np.sqrt((x[:,:,18,None] - x[:,:,24,None])**2 + (x[:,:,19,None] - x[:,:,25,None])**2 + (x[:,:,20,None] - x[:,:,26,None])**2)
        rAnkle_rHeel = np.sqrt((x[:,:,21,None] - x[:,:,27,None])**2 + (x[:,:,22,None] - x[:,:,28,None])**2 + (x[:,:,23,None] - x[:,:,29,None])**2)
        rHeel_rSmallToe = np.sqrt((x[:,:,27,None] - x[:,:,33,None])**2 + (x[:,:,28,None] - x[:,:,34,None])**2 + (x[:,:,29,None] - x[:,:,35,None])**2)
        rHeel_rBigToe = np.sqrt((x[:,:,27,None] - x[:,:,39,None])**2 + (x[:,:,28,None] - x[:,:,40,None])**2 + (x[:,:,29,None] - x[:,:,41,None])**2)
        lAnkle_lHeel = np.sqrt((x[:,:,24,None] - x[:,:,30,None])**2 + (x[:,:,25,None] - x[:,:,31,None])**2 + (x[:,:,26,None] - x[:,:,32,None])**2)
        lHeel_lSmallToe = np.sqrt((x[:,:,30,None] - x[:,:,36,None])**2 + (x[:,:,31,None] - x[:,:,37,None])**2 + (x[:,:,32,None] - x[:,:,38,None])**2)
        lHeel_lBigToe = np.sqrt((x[:,:,30,None] - x[:,:,42,None])**2 + (x[:,:,31,None] - x[:,:,43,None])**2 + (x[:,:,32,None] - x[:,:,44,None])**2)
        x = np.concatenate((x, neck_rShoulder,neck_lShoulder, rShoulder_lShoulder,rShoulder_rHip,lShoulder_lHip, rHip_lHip,rHip_rKnee,lHip_lKnee,rKnee_rAnkle,lKnee_lAnkle, rAnkle_rHeel, rHeel_rSmallToe, rHeel_rBigToe, lAnkle_lHeel, lHeel_lSmallToe, lHeel_lBigToe), axis=2)
        

        x = torch.from_numpy(x).to(device=device, dtype=torch.float32)
        y = torch.from_numpy(y).to(device=device, dtype=torch.float32)
    
        #print("x = ", x.shape)
        #print("y = ", y.shape)
        embeddings = in_layer(x)
        #print('embeddings = ', embeddings.shape)

        hidden_state = gpt2(inputs_embeds=embeddings).last_hidden_state
        #print('hidden_state = ', hidden_state.shape)
    
        a = out_layer(hidden_state)
        #print("a = ", a.shape)
        #a_relu = nn.functional.relu(a)
    
        output = out_layer2(a)
        #output = nn.functional.relu(output_inter)
    
        loss = RMSELoss(output,y)
        loss_plot = RMSELoss(output, y)
    
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if i % 100 == 0:
            print("Train RMSE loss after epoch:" + str(epoch) + ", batch: " + str(i) + " = ", loss_plot)

        running_loss_plot += loss_plot
    train_mse_plot = running_loss_plot / len(train_generator)
    print("Train RMSE loss after an epoch:" + str(epoch) + "(averaged across batches)  = ", train_mse_plot)
    
    with torch.no_grad():
        running_loss_eval_plot = 0.0
        for i_eval in range(len(val_generator)):
            data_eval = val_generator[i_eval]
            x_eval = (data_eval[0])
            y_eval  = (data_eval[1])


            neck_rShoulder = np.sqrt((x_eval[:,:,0,None] - x_eval[:,:,3,None])**2 + (x_eval[:,:,1,None] - x_eval[:,:,4,None])**2 + (x_eval[:,:,2,None] - x_eval[:,:,5,None])**2)
            neck_lShoulder = np.sqrt((x_eval[:,:,0,None] - x_eval[:,:,6,None])**2 + (x_eval[:,:,1,None] - x_eval[:,:,7,None])**2 + (x_eval[:,:,2,None] - x_eval[:,:,8,None])**2)
            rShoulder_lShoulder = np.sqrt((x_eval[:,:,3,None] - x_eval[:,:,6,None])**2 + (x_eval[:,:,4,None] - x_eval[:,:,7,None])**2 + (x_eval[:,:,5,None] - x_eval[:,:,8,None])**2)
            rShoulder_rHip = np.sqrt((x_eval[:,:,3,None] - x_eval[:,:,9,None])**2 + (x_eval[:,:,4,None] - x_eval[:,:,10,None])**2 + (x_eval[:,:,5,None] - x_eval[:,:,11,None])**2)
            lShoulder_lHip = np.sqrt((x_eval[:,:,6,None] - x_eval[:,:,12,None])**2 + (x_eval[:,:,7,None] - x_eval[:,:,13,None])**2 + (x_eval[:,:,8,None] - x_eval[:,:,14,None])**2)
            rHip_lHip = np.sqrt((x_eval[:,:,9,None] - x_eval[:,:,12,None])**2 + (x_eval[:,:,10,None] - x_eval[:,:,13,None])**2 + (x_eval[:,:,11,None] - x_eval[:,:,14,None])**2)
            rHip_rKnee = np.sqrt((x_eval[:,:,9,None] - x_eval[:,:,15,None])**2 + (x_eval[:,:,10,None] - x_eval[:,:,16,None])**2 + (x_eval[:,:,11,None] - x_eval[:,:,17,None])**2)
            lHip_lKnee = np.sqrt((x_eval[:,:,12,None] - x_eval[:,:,18,None])**2 + (x_eval[:,:,13,None] - x_eval[:,:,19,None])**2 + (x_eval[:,:,14,None] - x_eval[:,:,20,None])**2)
            rKnee_rAnkle = np.sqrt((x_eval[:,:,15,None] - x_eval[:,:,21,None])**2 + (x_eval[:,:,16,None] - x_eval[:,:,22,None])**2 + (x_eval[:,:,17,None] - x_eval[:,:,23,None])**2)
            lKnee_lAnkle = np.sqrt((x_eval[:,:,18,None] - x_eval[:,:,24,None])**2 + (x_eval[:,:,19,None] - x_eval[:,:,25,None])**2 + (x_eval[:,:,20,None] - x_eval[:,:,26,None])**2)
            rAnkle_rHeel = np.sqrt((x_eval[:,:,21,None] - x_eval[:,:,27,None])**2 + (x_eval[:,:,22,None] - x_eval[:,:,28,None])**2 + (x_eval[:,:,23,None] - x_eval[:,:,29,None])**2)
            rHeel_rSmallToe = np.sqrt((x_eval[:,:,27,None] - x_eval[:,:,33,None])**2 + (x_eval[:,:,28,None] - x_eval[:,:,34,None])**2 + (x_eval[:,:,29,None] - x_eval[:,:,35,None])**2)
            rHeel_rBigToe = np.sqrt((x_eval[:,:,27,None] - x_eval[:,:,39,None])**2 + (x_eval[:,:,28,None] - x_eval[:,:,40,None])**2 + (x_eval[:,:,29,None] - x_eval[:,:,41,None])**2)
            lAnkle_lHeel = np.sqrt((x_eval[:,:,24,None] - x_eval[:,:,30,None])**2 + (x_eval[:,:,25,None] - x_eval[:,:,31,None])**2 + (x_eval[:,:,26,None] - x_eval[:,:,32,None])**2)
            lHeel_lSmallToe = np.sqrt((x_eval[:,:,30,None] - x_eval[:,:,36,None])**2 + (x_eval[:,:,31,None] - x_eval[:,:,37,None])**2 + (x_eval[:,:,32,None] - x_eval[:,:,38,None])**2)
            lHeel_lBigToe = np.sqrt((x_eval[:,:,30,None] - x_eval[:,:,42,None])**2 + (x_eval[:,:,31,None] - x_eval[:,:,43,None])**2 + (x_eval[:,:,32,None] - x_eval[:,:,44,None])**2)
            x_eval = np.concatenate((x_eval, neck_rShoulder,neck_lShoulder, rShoulder_lShoulder,rShoulder_rHip,lShoulder_lHip, rHip_lHip,rHip_rKnee,lHip_lKnee,rKnee_rAnkle,lKnee_lAnkle, rAnkle_rHeel, rHeel_rSmallToe, rHeel_rBigToe, lAnkle_lHeel, lHeel_lSmallToe, lHeel_lBigToe), axis=2)
            


            x_eval  = torch.from_numpy(x_eval).to(device=device, dtype=torch.float32)
            y_eval  = torch.from_numpy(y_eval).to(device=device, dtype=torch.float32)
    
            #print("x = ", x.shape)
            #print("y = ", y.shape)
            embeddings_eval = in_layer(x_eval)
            #print('embeddings = ', embeddings.shape)

            hidden_state_eval = gpt2(inputs_embeds=embeddings_eval).last_hidden_state
            #print('hidden_state = ', hidden_state.shape)
    
            a_eval = out_layer(hidden_state_eval)
            #print("a = ", a.shape)
           # a_eval_relu = nn.functional.relu(a_eval)
            
            output_eval  = out_layer2(a_eval)
            #output_eval = nn.functional.relu(output_inter_eval)
    
            loss_eval_plot  = RMSELoss(output_eval, y_eval)
            
            if i_eval % 10 == 0:
                print("Eval RMSE loss after epoch:" + str(epoch) + ", batch: " + str(i_eval )+  " = ", loss_eval_plot)

            running_loss_eval_plot += loss_eval_plot 
        eval_mse_plot = running_loss_eval_plot / len(val_generator)
        print(f"epoch {epoch + 1:2d}, train loss: {train_mse_plot:.8f}, eval loss {eval_mse_plot:.8f}")

    train_generator.on_epoch_end()
    val_generator.on_epoch_end()

