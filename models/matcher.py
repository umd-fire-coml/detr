# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Modules to compute the matching cost and solve the corresponding LSAP.
"""
import torch
from scipy.optimize import linear_sum_assignment
from torch import nn

from util.box_ops import box_cxcywh_to_xyxy, generalized_box_iou


class HungarianMatcher(nn.Module):
    """This class computes an assignment between the targets and the predictions of the network

    For efficiency reasons, the targets don't include the no_object. Because of this, in general,
    there are more predictions than targets. In this case, we do a 1-to-1 matching of the best predictions,
    while the others are un-matched (and thus treated as non-objects).
    """

    def __init__(self, cost_class: float = 1, cost_bbox: float = 1, cost_giou: float = 1):
        """Creates the matcher

        Params:
            cost_class: This is the relative weight of the classification error in the matching cost
            cost_bbox: This is the relative weight of the L1 error of the bounding box coordinates in the matching cost
            cost_giou: This is the relative weight of the giou loss of the bounding box in the matching cost
        """
        super().__init__()
        self.cost_class = cost_class
        self.cost_bbox = cost_bbox
        self.cost_giou = cost_giou
        assert cost_class != 0 or cost_bbox != 0 or cost_giou != 0, "all costs cant be 0"

    @torch.no_grad()
    def forward(self, outputs, targets):
        """ Performs the matching

        Params:
            outputs: This is a dict that contains at least these entries:
                 "pred_logits": Tensor of dim [batch_size, num_queries, num_classes] with the classification logits
                 "pred_boxes": Tensor of dim [batch_size, num_queries, 4] with the predicted box coordinates

            targets: This is a list of targets (len(targets) = batch_size), where each target is a dict containing:
                 "labels": Tensor of dim [num_target_boxes] (where num_target_boxes is the number of ground-truth
                           objects in the target) containing the class labels
                 "boxes": Tensor of dim [num_target_boxes, 4] containing the target box coordinates

        Returns:
            A list of size batch_size, containing tuples of (index_i, index_j) where:
                - index_i is the indices of the selected predictions (in order)
                - index_j is the indices of the corresponding selected targets (in order)
            For each batch element, it holds:
                len(index_i) = len(index_j) = min(num_queries, num_target_boxes)
        """
        # get the size of first two output dimensions, see Params
        bs, num_queries = outputs["pred_logits"].shape[:2]

        # We flatten to compute the cost matrices in a batch
        # outputs["pred_logits"] gets reshaped to [batch_size * num_queries, num_classes],
        # then apply softmax on the num_classes dimension
        out_prob = outputs["pred_logits"].flatten(0, 1).softmax(-1) 
        # outputs["pred_boxes"] gets reshaped to [batch_size * num_queries, 4]
        out_bbox = outputs["pred_boxes"].flatten(0, 1)  

        # Also concat the target labels and boxes
        # tgt_ids is a long list of labels for each bounding box, of shape [batch_size * num_target_boxes, 1]
        tgt_ids = torch.cat([img["labels"] for img in targets])
        # tgt_bbox is a long list of boxes for each bounding box, of shape [batch_size * num_target_boxes, 4]
        tgt_bbox = torch.cat([v["boxes"] for v in targets])

        # Compute the classification cost. Contrary to the loss, we don't use the NLL,
        # but approximate it in 1 - proba[target class].
        # The 1 is a constant that doesn't change the matching, it can be ommitted.
        # Select the class probability at num_classes[tgt_ids]
        # Higher out_prob gets lower cost
        # out_prob is a long list of class vector probabilities for each bounding box
        # cost_class shape: [batch_size * num_queries, len(tgt_ids)]
        # TODO: need intuitive, not wasting computation and memory, since only 1 out of bs will be used.
        #cost_class = -out_prob[:, tgt_ids]
        
        
        list_img_pred_probs = [probs for probs in outputs["pred_logits"]]
        list_img_target_classes = [img["labels"] for img in targets]
        list_img_cost_class = [
            self.cost_class * -queries_probs[:, target_classes]
            for queries_probs, target_classes in zip(list_img_pred_probs, list_img_target_classes)] 

        # Compute the L1 cost between boxes for each value in the bounding box
        list_img_pred_bboxes = [bboxes for bboxes in outputs["pred_boxes"]]
        list_img_target_bboxes = [img["boxes"] for img in targets]
        list_img_cost_bboxes = [
            self.cost_bbox * torch.cdist(pred_bboxes, target_bboxes, p=1)
            for pred_bboxes, target_bboxes in zip(list_img_pred_bboxes, list_img_target_bboxes)]
        #cost_bbox = torch.cdist(out_bbox, tgt_bbox, p=1)

        # Compute the giou cost betwen boxes
        list_img_cost_giou = [
            self.cost_giou * -generalized_box_iou(box_cxcywh_to_xyxy(pred_bboxes), box_cxcywh_to_xyxy(target_bboxes))
            for pred_bboxes, target_bboxes in zip(list_img_pred_bboxes, list_img_target_bboxes)]
        #cost_giou = -generalized_box_iou(box_cxcywh_to_xyxy(out_bbox), box_cxcywh_to_xyxy(tgt_bbox))
        
        # Final cost matrix
        list_img_cost_matrix = [
            cost_class + cost_bboxes + cost_giou
            for cost_class, cost_bboxes, cost_giou in zip(list_img_cost_class, list_img_cost_bboxes, list_img_cost_giou)]
        # Final cost matrix
        # C = self.cost_bbox * cost_bbox + self.cost_class * cost_class + self.cost_giou * cost_giou
        # reshape the cost of each bounding box back into shape [batch_size, num_queries, total_num_targets_in_img_batch]
        # C = C.view(bs, num_queries, -1).cpu()
        
        # get best matching indices
        list_img_matching_query_target = [linear_sum_assignment(cost_matrix) for cost_matrix in list_img_cost_matrix]
        return [(torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64)) for i, j in list_img_matching_query_target]

        # get the number of boxes of each image in the ground truth
        # num_targets_in_each_img = [len(img["boxes"]) for img in targets]
        
        # we need a list of cost matrix of each img of shape [num_queries, targets[i]]
        
        # C.shape is [bs, num_queries, sum(num_targets_in_each_img)] and c.shape is [bs, num_queries, num_targets_in_imgs[i]]
        # c[i].shape is [num_queries, sizes[i]]
        # indices is a list of tuples (row_ind, col_ind) of length batch_size, each representing one image
        # where row_ind is a list ids of length num_targets, col_ind is a list ids of length num_targets
        # indices = [linear_sum_assignment(c[i]) for i, c in enumerate(C.split(num_targets_in_imgs, -1))]
        
        # convert numpy array into torch tensor
        # return [(torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64)) for i, j in indices]


def build_matcher(args):
    return HungarianMatcher(cost_class=args.set_cost_class, cost_bbox=args.set_cost_bbox, cost_giou=args.set_cost_giou)
