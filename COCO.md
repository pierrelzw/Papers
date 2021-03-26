## COCO
### Dataset

### Evaluation
1. Unless otherwise specified, AP and AR are averaged over multiple Intersection over Union (IoU) values. Specifically we use 10 IoU thresholds of .50:.05:.95. This is a break from tradition, where AP is computed at a single IoU of .50 (which corresponds to our metric APIoU=.50). Averaging over IoUs rewards detectors with better localization.

	**在COCO中，AP通过平均10个iou（0.5:0.05:0.95）计算得到, 传统的计算（pascal_voc）只计算iou=0.5**

2. AP is averaged over all categories. Traditionally, this is called "mean average precision" (mAP). We make no distinction between AP and mAP (and likewise AR and mAR) and assume the difference is clear from context.
AP (averaged across all 10 IoU thresholds and all 80 categories) will determine the challenge winner. This should be considered the single most important metric when considering performance on COCO.

	**AP即mAP，是通过计算所有类别的AP得到**

3. In COCO, there are more small objects than large objects. Specifically: approximately 41% of objects are small (area < 322), 34% are medium (322 < area < 962), and 24% are large (area > 962). Area is measured as the number of pixels in the segmentation mask.
	**COCO数据集中，小目标多于大目标**
	
4. AR is the maximum recall given a fixed number of detections per image, averaged over categories and IoUs. AR is related to the metric of the same name used in proposal evaluation but is computed on a per-category basis.All metrics are computed allowing for at most 100 top-scoring detections per image (across all categories).
	
	**所有的metric是基于score前100的检测结果计算得到**

5. The evaluation metrics for detection with bounding boxes and segmentation masks are identical in all respects except for the IoU computation (which is performed over boxes or masks, respectively).
	**检测和分割的mask是一样的，只是计算IoU的方式不同**