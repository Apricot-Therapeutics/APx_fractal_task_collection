### Purpose
- Segments secondary objects based on primary labels and intensity images using watershed segmentation.  
- Supports multiplexed and non-multiplexed acquisitions.  
- Optional parameters allow for flexible handling of thresholds, blurring, and masking.  

### Limitations
- Requires consistent **label and channel names** across input zarrs.  
- Assumes **registered well ROI tables** and NGFF-compatible metadata.  
- Only tested for **level 0 resolution**.  