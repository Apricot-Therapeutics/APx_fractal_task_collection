### Purpose
- Calculates pixel intensity correlation between two channels for each object in a label image.  
- Supports loading channels from different multiplexing acquisitions.  
- Handles multiple channel pairs per well and large datasets efficiently.  

### Outputs
- A **feature table** in the OME-Zarr structure with correlation values for specified channel pairs for each object.  

### Limitations
- Requires consistent **label and channel names** across input zarrs.  
- Assumes NGFF-compatible metadata.  
- Only validated for **level 0 resolution**.