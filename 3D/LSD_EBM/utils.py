import numpy as np
from scipy import ndimage
from hausdorff import hausdorff_distance
from sklearn.metrics.cluster import normalized_mutual_info_score
from sklearn.metrics import cohen_kappa_score
from scipy.spatial.distance import directed_hausdorff
import point_cloud_utils as pcu

#-----------------------------------------------------#
#            Calculate : Confusion Matrix             #
#-----------------------------------------------------#
def calc_ConfusionMatrix(truth, pred, c=1, dtype=np.int64, **kwargs):
    # Obtain predicted and actual condition
    gt = np.equal(truth, c)
    pd = np.equal(pred, c)
    not_gt = np.logical_not(gt)
    not_pd = np.logical_not(pd)
    # Compute Confusion Matrix
    tp = np.logical_and(pd, gt).sum()
    tn = np.logical_and(not_pd, not_gt).sum()
    fp = np.logical_and(pd, not_gt).sum()
    fn = np.logical_and(not_pd, gt).sum()
    # Convert to desired numpy type to avoid overflow
    tp = tp.astype(dtype)
    tn = tn.astype(dtype)
    fp = fp.astype(dtype)
    fn = fn.astype(dtype)
    # Return Confusion Matrix
    return tp, tn, fp, fn

#-----------------------------------------------------#
#              Calculate : True Positive              #
#-----------------------------------------------------#
def calc_TruePositive(truth, pred, c=1, **kwargs):
    # Obtain predicted and actual condition
    gt = np.equal(truth, c)
    pd = np.equal(pred, c)
    not_gt = np.logical_not(gt)
    not_pd = np.logical_not(pd)
    # Compute true positive
    tp = np.logical_and(pd, gt).sum()
    # Return true positive
    return tp

#-----------------------------------------------------#
#              Calculate : True Negative              #
#-----------------------------------------------------#
def calc_TrueNegative(truth, pred, c=1, **kwargs):
    # Obtain predicted and actual condition
    gt = np.equal(truth, c)
    pd = np.equal(pred, c)
    not_gt = np.logical_not(gt)
    not_pd = np.logical_not(pd)
    # Compute true negative
    tn = np.logical_and(not_pd, not_gt).sum()
    # Return true negative
    return tn

#-----------------------------------------------------#
#              Calculate : False Positive             #
#-----------------------------------------------------#
def calc_FalsePositive(truth, pred, c=1, **kwargs):
    # Obtain predicted and actual condition
    gt = np.equal(truth, c)
    pd = np.equal(pred, c)
    not_gt = np.logical_not(gt)
    not_pd = np.logical_not(pd)
    # Compute false positive
    fp = np.logical_and(pd, not_gt).sum()
    # Return false positive
    return fp

#-----------------------------------------------------#
#              Calculate : False Negative             #
#-----------------------------------------------------#
def calc_FalseNegative(truth, pred, c=1, **kwargs):
    # Obtain predicted and actual condition
    gt = np.equal(truth, c)
    pd = np.equal(pred, c)
    not_gt = np.logical_not(gt)
    not_pd = np.logical_not(pd)
    # Compute false negative
    fn = np.logical_and(not_pd, gt).sum()
    # Return false negative
    return 

#-----------------------------------------------------#
#          Calculate : Volumetric Similarity          #
#-----------------------------------------------------#
def calc_VolumetricSimilarity(truth, pred, c=1, **kwargs):
    # Obtain confusion mat
    tp, tn, fp, fn = calc_ConfusionMatrix(truth.cpu().numpy(), pred.cpu().numpy(), c)
    # Compute VS
    if (2*tp + fp + fn) != 0:
        vs = 1 - (np.abs(fn-fp) / (2*tp + fp + fn))
    else : vs = 1.0 - 0.0
    # Return VS score
    return vs

#-----------------------------------------------------#
#        Calculate : Simple Hausdorff Distance        #
#-----------------------------------------------------#
def calc_SimpleHausdorffDistance(truth, pred, c=1, **kwargs):
    # Compute simple Hausdorff Distance
    hd = hausdorff_distance(truth.cpu().numpy(), pred.cpu().numpy(), distance="euclidean")
    # Return Hausdorff Distance
    return np.float64(hd)

#-----------------------------------------------------#
#       Calculate : Hausdorff Distance                #
#-----------------------------------------------------#
def calc_Hausdorff_Distance(truth, pred):
    return pcu.hausdorff_distance(truth.cpu().numpy(), pred.cpu().numpy())

#-----------------------------------------------------#
#       Calculate : Average Hausdorff Distance        #
#-----------------------------------------------------#
def border_map(binary_img):
    """
    Creates the border for a 3D or 2D image
    """
    ndims = binary_img.ndim
    binary_map = np.asarray(binary_img, dtype=np.uint8)

    if ndims == 2:
        left = ndimage.shift(binary_map, [-1, 0], order=0)
        right = ndimage.shift(binary_map, [1, 0], order=0)
        superior = ndimage.shift(binary_map, [0, 1], order=0)
        inferior = ndimage.shift(binary_map, [0, -1], order=0)
        cumulative = left + right + superior + inferior
        ndir = 4
    elif ndims == 3:
        left = ndimage.shift(binary_map, [-1, 0, 0], order=0)
        right = ndimage.shift(binary_map, [1, 0, 0], order=0)
        anterior = ndimage.shift(binary_map, [0, 1, 0], order=0)
        posterior = ndimage.shift(binary_map, [0, -1, 0], order=0)
        superior = ndimage.shift(binary_map, [0, 0, 1], order=0)
        inferior = ndimage.shift(binary_map, [0, 0, -1], order=0)
        cumulative = left + right + anterior + posterior + superior + inferior
        ndir = 6
    else:
        raise RuntimeError(f'Image must be of 2 or 3 dimensions, got {ndims}')

    border = ((cumulative < ndir) * binary_map) == 1
    return border

def border_distance(ref,seg):
    """
    This functions determines the map of distance from the borders of the
    segmentation and the reference and the border maps themselves
    """
    border_ref = border_map(ref)
    border_seg = border_map(seg)
    oppose_ref = 1 - ref
    oppose_seg = 1 - seg
    # euclidean distance transform
    distance_ref = ndimage.distance_transform_edt(oppose_ref)
    distance_seg = ndimage.distance_transform_edt(oppose_seg)
    distance_border_seg = border_ref * distance_seg
    distance_border_ref = border_seg * distance_ref
    return distance_border_ref, distance_border_seg

def calc_AverageHausdorffDistance(truth, pred, c=1, **kwargs):
    """
    This functions calculates the average symmetric distance and the
    hausdorff distance between a segmentation and a reference image
    :return: hausdorff distance and average symmetric distance
    """
    # Obtain sets with associated class
    ref = np.equal(truth.cpu().numpy(), c)
    seg = np.equal(pred.cpu().numpy(), c)
    # Compute AHD
    ref_border_dist, seg_border_dist = border_distance(ref, seg)
    hausdorff_distance = np.max([np.max(ref_border_dist),
                                 np.max(seg_border_dist)])
    # Return AHD
    return hausdorff_distance

#-----------------------------------------------------#
#           Calculate : Sensitivity via Sets          #
#-----------------------------------------------------#
def calc_Sensitivity_Sets(truth, pred, c=1, **kwargs):
    # Obtain sets with associated class
    gt = np.equal(truth.cpu().numpy(), c)
    pd = np.equal(pred.cpu().numpy(), c)
    # Calculate sensitivity
    if gt.sum() != 0 : sens = np.logical_and(pd, gt).sum() / gt.sum()
    else : sens = 0.0
    # Return sensitivity
    return sens

#-----------------------------------------------------#
#           Calculate : Specificity via Sets          #
#-----------------------------------------------------#
def calc_Specificity_Sets(truth, pred, c=1, **kwargs):
    # Obtain sets with associated class
    not_gt = np.logical_not(np.equal(truth.cpu().numpy(), c))
    not_pd = np.logical_not(np.equal(pred.cpu().numpy(), c))
    # Calculate specificity
    if (not_gt).sum() != 0:
        spec = np.logical_and(not_pd, not_gt).sum() / (not_gt).sum()
    else : spec = 0.0
    # Return specificity
    return spec

#-----------------------------------------------------#
#    Calculate : Normalized Mutual Information (NMI)  #
#-----------------------------------------------------#
def calc_Normalized_Mutual_Information(truth, pred):
    return normalized_mutual_info_score(truth.cpu().numpy().reshape(-1), pred.cpu().numpy().reshape(-1))

#-----------------------------------------------------#
#    Calculate : Cohen’s kappa                        #
#-----------------------------------------------------#
def calc_Cohen_kappa(truth, pred):
    return cohen_kappa_score(truth.cpu().numpy().reshape(-1), pred.cpu().numpy().reshape(-1))