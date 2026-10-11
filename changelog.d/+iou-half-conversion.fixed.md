`mean_iou_bbox` promotes half-precision boxes before converting xywh/cxcywh coordinates, so a small positive width at a large position does not round to a zero-width invalid box.
