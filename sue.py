import math
import pickle
import numpy as np
from tqdm import tqdm
import os
from sklearn.metrics import precision_recall_curve, auc

############# Loading precomputed data ################
with open(('st_luciatest_cvprcameraready.pickle'), 'rb') as handle: 
    # Precomputed data from STUN, BTL, SIFT, DELF, SuperPoint. All of them are already open-source, please refer to their corresponding publications/codes for implementations.
    # SUE only needs query descriptors, retrieved matches, and reference poses, and implementation is given later. 
    print('Loading precomputed data')
    q_mu = pickle.load(handle) # mean of query descriptors from STUN
    db_mu = pickle.load(handle) # mean of reference descriptors from STUN
    q_sigma_sq = pickle.load(handle) # variance of query descriptors from STUN
    db_sigma_sq = pickle.load(handle) # variance of reference descriptors from STUN
    preds = pickle.load(handle) # Top-10 reference matches for each query with STUN backbone
    dists = pickle.load(handle) # L2-distances of Top-10 reference matches for each query with STUN backbone
    gt = pickle.load(handle) # ground-truth reference matches for each query
    _ = pickle.load(handle)
    _ = pickle.load(handle)
    _ = pickle.load(handle)
    ref_poses = pickle.load(handle) # UTM poses of all reference descriptors 
    delf_inliers = pickle.load(handle) # inlier scores for all queries from DELF-RANSAC
    sift_inliers = pickle.load(handle) # inlier scores for all queries from SIFT-RANSAC
    superpoint_inliers = pickle.load(handle) # inlier scores for all queries from Superpoint-RANSAC
    preds_btl = pickle.load(handle) # Top-10 reference matches for each query from BTL. Note BTL has a different backbone than STUN, thus the different retrieved references.
    dists_btl = pickle.load(handle) # L2-distances of Top-10 reference matches for each query with BTL backbone
    q_mu_btl = pickle.load(handle) # mean of query descriptors from BTL 
    db_mu_btl = pickle.load(handle) # mean of reference descriptors from BTL
    q_sigma_sq_btl = pickle.load(handle) # variance of query descriptors from BTL
    db_sigma_sq_btl = pickle.load(handle) # variance of reference descriptors from BTL
    print('Done!')
########################################################


total_queries = len(preds)
q_sigma_sq_h = np.mean(q_sigma_sq, axis=1) # Averaging since STUN outputs variance for each dimension of the N-dimensional descriptor. Note, unlike STUN, BTL outputs single for variance for all dimensions, i.e., assuming Homoscedastic Gaussian noise added to feature dimensions
matched_array_foraucpr = np.zeros(total_queries, dtype="float32")
l2distances = np.zeros(total_queries)
sigmas = np.zeros(total_queries)
matched_array_foraucpr_btl = np.zeros(len(preds_btl), dtype="float32")
l2distances_btl = np.zeros(len(preds_btl))
sigmas_btl = np.zeros(len(preds_btl))

############# Checking if the retrieved indices are ground-truth matches, both for STUN and BTL computed descriptors ################
for itr, pred in enumerate(preds):
    if (np.any(np.in1d(pred[:1], gt[itr]))):  #checking if Top-1 contains GT
        matched_array_foraucpr[itr]=1.0
        l2distances[itr] = dists[itr][0]
        sigmas[itr] = q_sigma_sq_h[itr]

    else:
        matched_array_foraucpr[itr]=0.0
        l2distances[itr] = dists[itr][0]
        sigmas[itr] = q_sigma_sq_h[itr]

for itr, pred_btl in enumerate(preds_btl):
    if (np.any(np.in1d(pred_btl[:1], gt[itr]))):  #checking if Top-1 contains GT
        matched_array_foraucpr_btl[itr]=1.0
        l2distances_btl[itr] = dists_btl[itr][0]
        sigmas_btl[itr] = q_sigma_sq_btl[itr][0]

    else:
        matched_array_foraucpr_btl[itr]=0.0
        l2distances_btl[itr] = dists_btl[itr][0]
        sigmas_btl[itr] = q_sigma_sq_btl[itr][0]
########################################################

############# Formatting all the baselines into appropriate confidences ################
match_scores = -1 * l2distances
match_scores = np.interp(match_scores, (match_scores.min(), match_scores.max()), (0.1, 1.0))

pa_scores = np.zeros(total_queries)
for itr in range(total_queries):
    pa_scores[itr] =  dists[itr][0] / dists[itr][1]  
    
pa_scores = -1 * pa_scores    
pa_scores = np.interp(pa_scores, (pa_scores.min(), pa_scores.max()), (0.1, 1.0))

stun_scores = -1 * sigmas
stun_scores = np.interp(stun_scores, (stun_scores.min(), stun_scores.max()), (0.1, 1.0))

btl_scores = -1 * sigmas_btl
btl_scores = np.interp(btl_scores, (btl_scores.min(), btl_scores.max()), (0.1, 1.0))
    
delf_ransac_scores = np.interp(delf_inliers, (delf_inliers.min(), delf_inliers.max()), (0.1, 1.0))
sift_ransac_scores = np.interp(sift_inliers, (sift_inliers.min(), sift_inliers.max()), (0.1, 1.0))
superpoint_ransac_scores = np.interp(superpoint_inliers, (superpoint_inliers.min(), superpoint_inliers.max()), (0.1, 1.0))
########################################################

############# SUE starts ################
sue_scores = np.zeros(total_queries)
num_NN = 10 # Number of nearest neighbours 
slope = 350 # Slope hyper-parameter of the Gaussian used to down-weight the contributions of nearest neighbours 

print('Computing SUE uncertainty')    
weights = np.ones(num_NN)
for itr in tqdm(range(len(sue_scores))):   
    top_preds = preds[itr][:num_NN]
    nn_poses = ref_poses[top_preds]
    bm_pose = nn_poses[0]

    for itr2 in range(num_NN):
        weights[itr2] = math.e ** ((-1*abs(dists[itr][itr2])) * slope) 

    weights = weights/sum(abs(weights))

    mean_pose = np.asarray([np.average(nn_poses[:,0], weights=weights), np.average(nn_poses[:,1], weights=weights)])

    variance_lat_lat = 0 
    variance_lon_lon = 0    
    variance_lat_lon = 0    

    for k in range(0, num_NN):                
        diff_lat_lat = min(500, nn_poses[k,0] - mean_pose[0]) # so everything that is more than 500 meters away contributes equally to the variance 
        diff_lon_lon = min(500, nn_poses[k,1] - mean_pose[1])
        diff_lat_lon = min(500, nn_poses[k,0] - mean_pose[0]) *  min(500, nn_poses[k,1] - mean_pose[1])
                
        variance_lat_lat = variance_lat_lat + weights[k] * (diff_lat_lat)**2
        variance_lon_lon = variance_lon_lon + weights[k] * (diff_lon_lon)**2
        variance_lat_lon = variance_lat_lon + weights[k] * diff_lat_lon
        
    sue_scores[itr] = (variance_lat_lat + variance_lon_lon)/2  # assuming independent dimensions
  
sue_scores = -1 * sue_scores # converting into a confidence instead of an uncertainty
sue_scores_normalized = np.interp(sue_scores, (sue_scores.min(), sue_scores.max()), (0.0, 0.9999)) # avoiding infinity
print('Done!') 
############# SUE ends. Just like that. ################

############# Computing AUC-PR values ################
precision_basedonmatchscore, recall_basedonmatchscore, _ = precision_recall_curve(matched_array_foraucpr, match_scores)
precision_basedonstunscore, recall_basedonstunscore, _ = precision_recall_curve(matched_array_foraucpr, stun_scores)
precision_basedonbtlscore, recall_basedonbtlscore, _ = precision_recall_curve(matched_array_foraucpr_btl, btl_scores)
precision_basedonpascore, recall_basedonpascore, _ = precision_recall_curve(matched_array_foraucpr, pa_scores)
precision_basedondelfscore, recall_basedondelfscore, _ = precision_recall_curve(matched_array_foraucpr, delf_ransac_scores)
precision_basedonsiftscore, recall_basedonsiftscore, _ = precision_recall_curve(matched_array_foraucpr, sift_ransac_scores)
precision_basedonsuperpointscore, recall_basedonsuperpointscore, _ = precision_recall_curve(matched_array_foraucpr, superpoint_ransac_scores)
precision_basedonsuescore, recall_basedonsuescore, _ = precision_recall_curve(matched_array_foraucpr, sue_scores_normalized)

auc_basedonmatchscore = auc(recall_basedonmatchscore, precision_basedonmatchscore)
auc_basedonstunscore = auc(recall_basedonstunscore, precision_basedonstunscore)
auc_basedonbtlscore = auc(recall_basedonbtlscore, precision_basedonbtlscore)
auc_basedonpascore = auc(recall_basedonpascore, precision_basedonpascore)
auc_basedondelfscore = auc(recall_basedondelfscore, precision_basedondelfscore)
auc_basedonsiftscore = auc(recall_basedonsiftscore, precision_basedonsiftscore)
auc_basedonsuperpointscore = auc(recall_basedonsuperpointscore, precision_basedonsuperpointscore)
auc_basedonsuescore = auc(recall_basedonsuescore, precision_basedonsuescore)

print('###########')
print('AUC-PR_based_on_L2-distance: ', auc_basedonmatchscore)
print('AUC-PR_based_on_PA-score: ', auc_basedonpascore)
print('AUC-PR_based_on_BTL: ', auc_basedonbtlscore)
print('AUC-PR_based_on_STUN: ', auc_basedonstunscore)
print('AUC-PR_based_on_SUE: ', auc_basedonsuescore)
print('###########')
print('AUC-PR_based_on_SIFT-RANSAC: ', auc_basedonsiftscore)
print('AUC-PR_based_on_DELF-RANSAC: ', auc_basedondelfscore)
print('AUC-PR_based_on_SuperPoint-RANSAC: ', auc_basedonsuperpointscore)
print('###########')
############# EOF. Kept it simple as promised in our SUE paper ################
