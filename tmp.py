import numpy as np

def rpn_to_roi(rpn_layer, regr_layer, C, dim_ordering, use_regr=True, max_boxes=300,overlap_thresh=0.9):
    """Convert rpn layer to roi bboxes

    Args: (num_anchors = 9)
        rpn_layer: output layer for rpn classification 
            shape (1, feature_map.height, feature_map.width, num_anchors)
            Might be (1, 18, 25, 18) if resized image is 400 width and 300
        regr_layer: output layer for rpn regression
            shape (1, feature_map.height, feature_map.width, num_anchors)
            Might be (1, 18, 25, 72) if resized image is 400 width and 300
        C: config
        use_regr: Wether to use bboxes regression in rpn
        max_boxes: max bboxes number for non-max-suppression (NMS)
        overlap_thresh: If iou in NMS is larger than this threshold, drop the box

    Returns:
        result: boxes from non-max-suppression (shape=(300, 4))
            boxes: coordinates for bboxes (on the feature map)
    """
    regr_layer = regr_layer / C.std_scaling

    anchor_sizes = C.anchor_box_scales   # (3 in here)
    anchor_ratios = C.anchor_box_ratios  # (3 in here)

    assert rpn_layer.shape[0] == 1

    (rows, cols) = rpn_layer.shape[1:3]

    curr_layer = 0

    # A.shape = (4, feature_map.height, feature_map.width, num_anchors) 
    # Might be (4, 18, 25, 18) if resized image is 400 width and 300
    # A is the coordinates for 9 anchors for every point in the feature map 
    # => all 18x25x9=4050 anchors cooridnates
    A = np.zeros((4, rpn_layer.shape[1], rpn_layer.shape[2], rpn_layer.shape[3]))

    for anchor_size in anchor_sizes:
        for anchor_ratio in anchor_ratios:
            # anchor_x = (128 * 1) / 16 = 8  => width of current anchor
            # anchor_y = (128 * 2) / 16 = 16 => height of current anchor
            anchor_x = (anchor_size * anchor_ratio[0])/C.rpn_stride
            anchor_y = (anchor_size * anchor_ratio[1])/C.rpn_stride
            
            # curr_layer: 0~8 (9 anchors)
            # the Kth anchor of all position in the feature map (9th in total)
            regr = regr_layer[0, :, :, 4 * curr_layer:4 * curr_layer + 4] # shape => (18, 25, 4)
            regr = np.transpose(regr, (2, 0, 1)) # shape => (4, 18, 25)

            # Create 18x25 mesh grid
            # For every point in x, there are all the y points and vice versa
            # X.shape = (18, 25)
            # Y.shape = (18, 25)
            X, Y = np.meshgrid(np.arange(cols),np. arange(rows))

            # Calculate anchor position and size for each feature map point
            A[0, :, :, curr_layer] = X - anchor_x/2 # Top left x coordinate
            A[1, :, :, curr_layer] = Y - anchor_y/2 # Top left y coordinate
            A[2, :, :, curr_layer] = anchor_x       # width of current anchor
            A[3, :, :, curr_layer] = anchor_y       # height of current anchor

            # Apply regression to x, y, w and h if there is rpn regression layer
            if use_regr:
                A[:, :, :, curr_layer] = apply_regr_np(A[:, :, :, curr_layer], regr)

            # Avoid width and height exceeding 1
            A[2, :, :, curr_layer] = np.maximum(1, A[2, :, :, curr_layer])
            A[3, :, :, curr_layer] = np.maximum(1, A[3, :, :, curr_layer])

            # Convert (x, y , w, h) to (x1, y1, x2, y2)
            # x1, y1 is top left coordinate
            # x2, y2 is bottom right coordinate
            A[2, :, :, curr_layer] += A[0, :, :, curr_layer]
            A[3, :, :, curr_layer] += A[1, :, :, curr_layer]

            # Avoid bboxes drawn outside the feature map
            A[0, :, :, curr_layer] = np.maximum(0, A[0, :, :, curr_layer])
            A[1, :, :, curr_layer] = np.maximum(0, A[1, :, :, curr_layer])
            A[2, :, :, curr_layer] = np.minimum(cols-1, A[2, :, :, curr_layer])
            A[3, :, :, curr_layer] = np.minimum(rows-1, A[3, :, :, curr_layer])

            curr_layer += 1

    all_boxes = np.reshape(A.transpose((0, 3, 1, 2)), (4, -1)).transpose((1, 0))  # shape=(4050, 4)
    all_probs = rpn_layer.transpose((0, 3, 1, 2)).reshape((-1))                   # shape=(4050,)

    # Apply non_max_suppression
    # Only extract the bboxes. Don't need rpn probs in the later process
    #result = non_max_suppression_fast(all_boxes, all_probs, overlap_thresh=overlap_thresh, max_boxes=max_boxes)[0]
    nms_idx_1d, nms_boxes, nms_probs = non_max_suppression_fast(all_boxes, all_probs, overlap_thresh=overlap_thresh, max_boxes=max_boxes)

    nms_idx_A, nms_idx_H, nms_idx_W = np.unravel_index(nms_idx_1d, (rpn_layer.shape[3], rpn_layer.shape[1], rpn_layer.shape[2]))
    nms_idx_2d = np.column_stack((nms_idx_H, nms_idx_W, nms_idx_A))
    return A, nms_idx_2d, nms_boxes, nms_probs

def get_anchor_pos_neg_idx_nms(R_list, img_datas, vi_features, C):
    """get anchor idx, postive idx, negative idx
        Args : 
            R_list : list of R (len : num_cam)
                R: (all_boxes, nms_idx, nms_bboxes, nms_probs) (shape=(H, W, A, 4), shape=(300, 3), shape=(300,4), shape=(300,) )
            img_datas : GT box
            vi_features : view_invariant feature (shape: 1, num_cam, H, W, A, vi_featue_size)
        Output :
            anchor_target_idx : anchor_idx(cam1) and target_idx (shape: 1, 2, num_sample, 4, 1)
                first : num_batch
                second : 0 = anchor, 1 = target
                third : num_sample = camPerm * numGTbox
                fourth : (cam_idx, h_idx, w_idx, a_idx)
    """
    im_size, rpn_stride, vi_max_overlap = C.im_size, C.rpn_stride, C.vi_max_overlap
    width, height = img_datas[0]['width'], img_datas[0]['height']
    num_anchor = len(img_datas[0]['bboxes'])

    #anchor
    anchor_idx_list = -np.ones((C.num_cam, num_anchor, 4), dtype=int)
    nms_HWA_idx_list = np.zeros((C.num_cam, C.num_nms, 3), dtype=int) # HWA
    ious = []
    for cam_idx in range(C.num_cam) :
        gt_bboxes = img_datas[cam_idx]['bboxes']
        _, nms_HWA_idx, nms_bboxes, _ = R_list[cam_idx]
        anchor_idx, iou = get_anchor_idx_nms(nms_HWA_idx, nms_bboxes, gt_bboxes, width, height, im_size, rpn_stride, vi_max_overlap)
        cam_idx_array = np.repeat(cam_idx, num_anchor).reshape(-1, 1)
        anchor_idx_list[cam_idx] = np.concatenate([cam_idx_array, anchor_idx], 1) #(num_anchor, 4)
        nms_HWA_idx_list[cam_idx] = nms_HWA_idx #(num_nms, 3)
        ious.append(iou)

    # delete anchor if even one cam doesn't have that anchor.
    invalid_anchor = np.where(anchor_idx_list[:,:,1:] == [-1, -1, -1])[1] #(axis 1)
    anchor_idx_list = np.delete(anchor_idx_list, invalid_anchor, axis = 1) 
    if not anchor_idx_list.size : return np.zeros(0)

    #add pos
    cam_idx_list = np.arange(C.num_cam)
    cam_idx_perms = np.array(list(permutations(cam_idx_list, 2))) #(Perm_size, 2)
    anchor_pos_idx_list = anchor_idx_list[cam_idx_perms] #(Per_size, 2, 1, 4)
    anchor_pos_idx_list = np.transpose(np.squeeze(anchor_pos_idx_list), (1, 0, 2)) #(2, Per_size, 4)

    #add neg
    vi_features = vi_features[0] #(num_cam, H, W, A, vi_featue_size)

    anchor_idx, positive_idx = anchor_pos_idx_list
    anchor_CHWA_idx = tuple(anchor_idx.T)
    positive_CHWA_idx = tuple(positive_idx.T)
    anchor_feature = vi_features[anchor_CHWA_idx] #(numSample, feature_size)

    postive_feature = vi_features[positive_CHWA_idx] #(numSample, feature_size)
    pos_dist = calc_dist(anchor_feature, postive_feature) #(numSample, )

    negative_cam_idx = positive_CHWA_idx[0]
    nms_vi_features = get_nms_vi_features(vi_features, nms_HWA_idx_list)
    negative_cand_feature = nms_vi_features[negative_cam_idx] #(numSample, 300, vi_feature_size)
    negative_nms_idx = get_min_dist_idx(anchor_feature, negative_cand_feature, pos_dist)
    negative_HWA_idx = nms_HWA_idx_list[negative_cam_idx, negative_nms_idx] #(num_sample, 3)
    negative_CHWA_idx = np.concatenate([negative_cam_idx.reshape(-1, 1), negative_HWA_idx], 1) #(num_sample, 4)

    anchor_pos_neg_idx = np.concatenate([anchor_pos_idx_list, np.expand_dims(negative_CHWA_idx, 0)], 0)#(3, num_sample, 4)
    return anchor_pos_neg_idx[np.newaxis, :, :, :, np.newaxis, np.newaxis]

def reid300(vi_feats, R_list, C) :
    """Generate matched boxes based on epipolar having top nms pob in one camera.
    Args: (num_anchors = 9)
        R_list: [(all_box, nms_idx, nms_box, nms_prob)]*num_cam,
                #all_box = #(4, grid_H, grid_W, anchor=9)
                #nms_idx = #(300, 3(=H,W,A)) 
                #nms_box = #(300, 4)
                #nms_pob = #(300,)
        C: config
        debug_img: [debug_img]*num_cam

    Returns:
        result: [matched_box_list_in_cam0,  matched_box_list_in_cam1, ...]
            matched_box_list_in_cam0 : #(n, 4), n is the number of boxes
    """
    nms_HWA_idx_list = [R[1] for R in R_list]
    nms_box_list = [R[2] for R in R_list]
    nms_prob_list = [R[3] for R in R_list]
    
    vi_feats = vi_feats[0] #(num_cam, H, W, A, feat_size)
    nms_vi_feats_list = [cur_cam_vi_feats[tuple(nms_idx.T)] for cur_cam_vi_feats, nms_idx in zip(vi_feats, nms_HWA_idx_list)]

    cam_idx_list = np.arange(C.num_cam)
    cam_idx = np.repeat(cam_idx_list, C.num_nms).reshape(-1, 1)
    nms_box = np.concatenate(nms_box_list, 0)
    nms_vi_feats = np.concatenate(nms_vi_feats_list, 0) #(num_cam*num_nms, feat_size)
    nms_prob = np.concatenate(nms_prob_list, 0)
    result = np.concatenate([cam_idx, nms_box, nms_vi_feats], 1)

    idx_in_nms = np.tile(np.arange(300), 3).reshape(-1, 1)
    idx_of_cam_nms = np.concatenate([cam_idx, idx_in_nms], 1)
    top_N_idx_of_cam_nms = idx_of_cam_nms[nms_prob.argsort()[-C.num_nms:]] #(300, 2)

    #sorting
    top_N_result =  result[nms_prob.argsort()[-C.num_nms:]] #(num_nms, 8)
        
    #anchor
    anchor_cam_idx = top_N_result[:, 0].astype(int)
    anchor_box = top_N_result[:, 1:5].astype(int)
    anchor_vi_feats = top_N_result[:, 5:] #(num_nms, feat_size)
    
    nms_arange = np.arange(C.num_nms)
    reid_box = -np.ones((C.num_cam, C.num_nms, 4))
    reid_box[(anchor_cam_idx, nms_arange)] = anchor_box
    nms_box_np = np.stack(nms_box_list, 0) #(num_cam, num_nms, 4)
    nms_vi_features = get_nms_vi_features(vi_feats, nms_HWA_idx_list)

    for offset in range(1, C.num_cam):
        target_cam_idx = (anchor_cam_idx+offset) % C.num_cam
        target_feats = nms_vi_features[target_cam_idx] #(num_nms, num_nms, vi_feature_size)
        target_nms_idx = get_min_dist_idx(anchor_vi_feats, target_feats)
        matched_box = nms_box_np[(target_cam_idx, target_nms_idx)] #(300, 4)
        reid_box[(target_cam_idx, nms_arange)] = matched_box 
    reid_box_list = [reid_box_in_one_cam for reid_box_in_one_cam in reid_box]
    return reid_box_list, anchor_cam_idx
 
def apply_regr_np(X, T):
    """Apply regression layer to all anchors in one feature map

    Args:
        X: shape=(4, 18, 25) the current anchor type for all points in the feature map
        T: regression layer shape=(4, 18, 25)

    Returns:
        X: regressed position and size for current anchor
    """
    try:
        x = X[0, :, :]
        y = X[1, :, :]
        w = X[2, :, :]
        h = X[3, :, :]

        tx = T[0, :, :]
        ty = T[1, :, :]
        tw = T[2, :, :]
        th = T[3, :, :]

        cx = x + w/2.
        cy = y + h/2.
        cx1 = tx * w + cx
        cy1 = ty * h + cy

        w1 = np.exp(tw.astype(np.float64)) * w
        h1 = np.exp(th.astype(np.float64)) * h
        x1 = cx1 - w1/2.
        y1 = cy1 - h1/2.

        x1 = np.round(x1)
        y1 = np.round(y1)
        w1 = np.round(w1)
        h1 = np.round(h1)
        return np.stack([x1, y1, w1, h1])
    except Exception as e:
        print(e)
        return X
 
def non_max_suppression_fast(boxes, probs, overlap_thresh=0.9, max_boxes=300):
    # code used from here: http://www.pyimagesearch.com/2015/02/16/faster-non-maximum-suppression-python/
    # if there are no boxes, return an empty list

    # Process explanation:
    #   Step 1: Sort the probs list
    #   Step 2: Find the larget prob 'Last' in the list and save it to the pick list
    #   Step 3: Calculate the IoU with 'Last' box and other boxes in the list. If the IoU is larger than overlap_threshold, delete the box from list
    #   Step 4: Repeat step 2 and step 3 until there is no item in the probs list 
    if len(boxes) == 0:
        return []

    # grab the coordinates of the bounding boxes
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]

#    np.testing.assert_array_less(x1, x2)
#    np.testing.assert_array_less(y1, y2)

    # if the bounding boxes integers, convert them to floats --
    # this is important since we'll be doing a bunch of divisions
    if boxes.dtype.kind == "i":
        boxes = boxes.astype("float")

    # initialize the list of picked indexes 
    pick = []

    # calculate the areas
    area = (x2 - x1) * (y2 - y1)

    # sort the bounding boxes 
    idxs = np.argsort(probs)

    # sort the bounding boxes 
    invalid_idxs = np.where((x1[idxs] - x2[idxs] >= 0) | (y1[idxs] - y2[idxs] >= 0))
    idxs = np.delete(idxs, invalid_idxs, 0)
    np.testing.assert_array_less(x1[idxs], x2[idxs])
    np.testing.assert_array_less(y1[idxs], y2[idxs])

    # keep looping while some indexes still remain in the indexes
    # list
    while len(idxs) > 0:
        # grab the last index in the indexes list and add the
        # index value to the list of picked indexes
        last = len(idxs) - 1
        i = idxs[last]
        pick.append(i)
        # find the intersection

        xx1_int = np.maximum(x1[i], x1[idxs[:last]])
        yy1_int = np.maximum(y1[i], y1[idxs[:last]])
        xx2_int = np.minimum(x2[i], x2[idxs[:last]])
        yy2_int = np.minimum(y2[i], y2[idxs[:last]])

        ww_int = np.maximum(0, xx2_int - xx1_int)
        hh_int = np.maximum(0, yy2_int - yy1_int)

        area_int = ww_int * hh_int

        # find the union
        area_union = area[i] + area[idxs[:last]] - area_int

        # compute the ratio of overlap
        overlap = area_int/(area_union + 1e-6)

        # delete all indexes from the index list that have
        idxs = np.delete(idxs, np.concatenate(([last],
            np.where(overlap > overlap_thresh)[0])))

        if len(pick) >= max_boxes:
            break

    # return only the bounding boxes that were picked using the integer data type
    boxes_idx = pick
    boxes = boxes[pick].astype("int")
    probs = probs[pick]
    return boxes_idx, boxes, probs


