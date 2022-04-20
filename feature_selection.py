import numpy as np


def calc_distance(a, b):
    if a.ndim < 2:
        a = np.atleast_2d(a)
    if b.ndim < 2:
        b = np.atleast_2d(b)
    d = np.linalg.norm(a-b, axis=1)
    return d.squeeze()

def calc_intra_monomer_dbd_distance(coors, distlist, indices, start_frame):
    current_frame = coors.shape[0]
    count = 0

    # PROA
    for j in range(len(indices)-1):
        resid1 = indices[j] - 2
        for k in range(j+1, len(indices)):
            resid2 = indices[k] - 2
            res1_position = coors[:,:,resid1].squeeze()
            res2_position = coors[:,:,resid2].squeeze()
            distlist[start_frame:start_frame+current_frame, count] = calc_distance(res1_position, res2_position)
            count += 1
    # PROB
    for j in range(len(indices)-1):
        resid1 = indices[j] - 2 + 202
        for k in range(j+1, len(indices)):
            resid2 = indices[k] - 2 + 202
            res1_position = coors[:,:,resid1].squeeze()
            res2_position = coors[:,:,resid2].squeeze()
            distlist[start_frame:start_frame+current_frame, count] = calc_distance(res1_position, res2_position)
            count += 1
    return distlist

def calc_intra_monomer_lbd_distance(coors, distlist, indices, start_frame):
    current_frame = coors.shape[0]
    count = 0

    # PROA
    for j in range(len(indices)-1):
        # The last 3 residues are from the other monomer
        # We are calculating PROA distances here, so 'the other monomer' is PROB
        if j >= len(indices) - 3:
            resid1 = indices[j] - 2 + 202
        else:
            resid1 = indices[j] - 2
        for k in range(j+1, len(indices)):
            if k >= len(indices) - 3:
                resid2 = indices[k] - 2 + 202
            else:
                resid2 = indices[k] - 2
            res1_position = coors[:,:,resid1].squeeze()
            res2_position = coors[:,:,resid2].squeeze()
            distlist[start_frame:start_frame+current_frame, count] = calc_distance(res1_position, res2_position)
            count += 1
    # PROB
    for j in range(len(indices)-1):
        # The last 3 residues are from the other monomer
        # We are calculating PROB distances here, so 'the other monomer' is PROA
        if j >= len(indices) - 3:
            resid1 = indices[j] - 2
        else:
            resid1 = indices[j] - 2 + 202
        for k in range(j+1, len(indices)):
            if k >= len(indices) - 3:
                resid2 = indices[k] - 2
            else:
                resid2 = indices[k] - 2 + 202
            res1_position = coors[:,:,resid1].squeeze()
            res2_position = coors[:,:,resid2].squeeze()
            distlist[start_frame:start_frame+current_frame, count] = calc_distance(res1_position, res2_position)
            count += 1
    return distlist

def calc_intra_monomer_pairwise_distance(coors, distlist, indices, start_frame):
    current_frame = coors.shape[0]
    count = 0

    # PROA  
    for j in range(indices.shape[0]):
        resid1 = indices[j, 0] - 2
        resid2 = indices[j, 1] - 2
        res1_position = coors[:,:,resid1].squeeze()
        res2_position = coors[:,:,resid2].squeeze()
        distlist[start_frame:start_frame+current_frame, count] = calc_distance(res1_position, res2_position)
        count += 1
        
    # PROB
    for j in range(indices.shape[0]):
        resid1 = indices[j, 0] - 2 + 202
        resid2 = indices[j, 1] - 2 + 202
        res1_position = coors[:,:,resid1].squeeze()
        res2_position = coors[:,:,resid2].squeeze()
        distlist[start_frame:start_frame+current_frame, count] = calc_distance(res1_position, res2_position)
        count += 1
    return distlist

def calc_inter_monomer_distance(coors, distlist, indices, start_frame):
    current_frame = coors.shape[0]
    
    for l in range(indices.shape[0]):
        resid1 = indices[l,0] - 2
        resid2 = indices[l,1] - 2 + 202
        res1_position = coors[:,:,resid1].squeeze()
        res2_position = coors[:,:,resid2].squeeze()
        distlist[start_frame:start_frame+current_frame, l] = calc_distance(res1_position, res2_position)
    return distlist

def calc_inter_monomer_all_distance(coors, distlist, indices, start_frame):
    current_frame = coors.shape[0]
    count = 0

    # PROA-PROB
    for l in range(indices.shape[0]):
        resid1 = indices[l,0] - 2
        resid2 = indices[l,1] - 2 + 202
        res1_position = coors[:,:,resid1].squeeze()
        res2_position = coors[:,:,resid2].squeeze()
        distlist[start_frame:start_frame+current_frame, count] = calc_distance(res1_position, res2_position)
        count += 1

    # PROB-PROA
    for l in range(indices.shape[0]):
        resid1 = indices[l,1] - 2
        resid2 = indices[l,0] - 2 + 202
        res1_position = coors[:,:,resid1].squeeze()
        res2_position = coors[:,:,resid2].squeeze()
        distlist[start_frame:start_frame+current_frame, count] = calc_distance(res1_position, res2_position)
        count += 1
    return distlist
