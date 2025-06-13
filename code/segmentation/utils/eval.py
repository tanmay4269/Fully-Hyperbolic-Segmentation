def compute_iou(preds, targets, num_classes, ignore_index=255):
    """ Computes mean IoU across classes, excluding ignore_index """
    ious = []
    for cls in range(num_classes):
        if cls == ignore_index:
            continue
        pred_inds = (preds == cls)
        target_inds = (targets == cls)
        intersection = (pred_inds & target_inds).sum().item()
        union = (pred_inds | target_inds).sum().item()
        if union == 0:
            continue
        ious.append(intersection / union)
    return sum(ious) / len(ious) if ious else 0.0