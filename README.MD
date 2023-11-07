# DICOMPreprocessor

A DICOM preprocessing module that can perform the following operations:

* Load DICOM files
* Correct for BitsAllocated≠BitsStored representation
* Convert to Hounsfield units (HU)
* Perform windowing based on HU
* Clean MRI/CT image from irrelevant objects (patient table, emergency button)
* Crop and resize using `fit_size` and `pixel_spacing` strategies

Installation of requirements via conda:

```
conda env create  -f environment.yml

conda activate dicom
```

Usage:

```
from dicomprocessing import preprocess
from dicomprocessing.utils import save_png

prc = preprocess.DICOMPreprocessor()

img, dcm = prc('my_scan.dcm') # read, clean, and crop

WindowWidth = dcm.WindowWidth # WindowWidth for optimal png compression

save_png('my_scan.png', img, WindowWidth) # save as png
```

