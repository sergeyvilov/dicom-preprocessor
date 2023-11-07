import numpy as np

import pydicom
import cv2
import skimage

from dicomprocessing import utils

    
class DICOMPreprocessor():
    '''
    DICOM preprocessor
    
    Load, clean, scale, and resize DICOM files.
    
    Parameters
    ----------
        
    rm_small_obj : bool, default=True
        Whether to remove small objects outside the body cross-sectional area (e.g. patient table, emergency button)
        
    mask_HU_thr : float, default=-150
        HU threshold for masking, usually the lower edge of HU window
        
    min_body_area_mm2 : float, default=12000
        Minimal body cross-sectional area for cropping mask, mm2
        
    max_hole_area_mm2 : float, default=10000
        Maximum hole size to fill in cropping mask, mm2  
        
    min_width_crop : float, default=100
        Minimal image width after cropping to consider cropping successfull, mm

    min_height_crop : float, default=80
        Minimal image height after cropping to consider cropping successfull, mm
        
    resize_strategy : str, default='fit_size'
        How to treat the image after cropping
        If set to 'fit_size', stretch to `output_size` without preserving pixel dimensions.
        If set to 'fixed_spacing', stretch each pixel to `pixel_spacing`, then center the image on a background of size `output_size`.
        
    pixel_spacing : tuple, default=None
        Size of each pixel for `resize_strategy="fixed_spacing"`: (delta_y, delta_x), mm
        
    output_size : tuple, default=(256,256)
        Size of the output image: (x, y), px
        
    background_HU : float, default=-1000
        Background HU for `resize_strategy="fixed_spacing"`
        
    window : tuple, default=(50,400)
        Center and width of the Hounsfield units (HU) window
        If set to 'dicom', use WindowCenter and WindowWidth DICOM tags.
        
    Examples
    --------
    
    >>>from dicomprocessing import preprocess    
    >>>prc = preprocess.DICOMPreprocessor()
    >>>img, dcm = prc('my_scan.dcm') # read, clean, and crop
    '''
    
    def __init__(self, window=(50,400), rm_small_obj=True, 
                 resize_strategy='fit_size', pixel_spacing=None, output_size=(256,256), 
                 min_body_area_mm2 = 12000, max_hole_area_mm2 = 10000,
                 min_width_crop=150, min_height_crop=80, background_HU = -1000, mask_HU_thr=-150, 
                 debug=False):
    
        self.window = window
        self.rm_small_obj = rm_small_obj 
        self.background_HU = background_HU 
        self.mask_HU_thr = mask_HU_thr 
        self.output_size = output_size 
        self.pixel_spacing = pixel_spacing
        self.min_width_crop = min_width_crop 
        self.min_height_crop = min_height_crop 
        self.min_body_area_mm2 = min_body_area_mm2 
        self.max_hole_area_mm2 = max_hole_area_mm2
        self.resize_strategy = resize_strategy 
        self.debug = debug

    def standardize_pixel_array(self, dcm):
        '''
        Convert pixels values from BitsStored - bit representation to BitsAllocated - bit representation
        
        When PixelRepresentation=1, the leading (BitsAllocated - BitsStored) bits are replaced with 1s for negative pixel values. 
        This ensures that 2's complement pixel values stored in BitsStored - bit representation are properly converted to BitsAllocated - bit representation.
        
        source: https://www.kaggle.com/code/huiminglin/rsna-2023-abdomen-dicom-standardization
        '''

        pixel_array = dcm.pixel_array

        if dcm.PixelRepresentation == 1:
            bit_shift = dcm.BitsAllocated - dcm.BitsStored
            dtype = pixel_array.dtype 
            pixel_array = (pixel_array << bit_shift).astype(dtype) >>  bit_shift

        return pixel_array

    def __call__(self, filepath):
        '''
        Process a given DICOM scan.
        
        Parameters
        ----------
        
        filepath : str
            Path to .dcm image
        
        '''
        
        if isinstance(filepath,str):
            dcm = pydicom.dcmread(filepath)
        else:
            dcm = filepath

        img = self.standardize_pixel_array(dcm) # correct for BitsStored≠BitsAllocated mismatch
        
        img = img*dcm.RescaleSlope + dcm.RescaleIntercept # pixels to Hounsfield units
        
        img0 = img.copy()
        
        keep_mask = np.zeros(img0.shape) # placeholder in case rm_small_obj = False 
        
        if self.rm_small_obj:
            
            img, keep_mask = self.crop(dcm, img) 
            
            if img.shape[0]*dcm.PixelSpacing[0]<self.min_height_crop or img.shape[1]*dcm.PixelSpacing[1]<self.min_width_crop:
                img = np.ones((100,100))*self.background_HU # if image is too small after cropping, replace it with a 100x100 square
            
        if self.resize_strategy=='fit_size':
            #stretch image to the given size, don't preserve pixel spacing
            img = cv2.resize(img, dsize=self.output_size)
        elif self.resize_strategy=='fixed_spacing':
            #stretch image to get the desired pixel size in mm, then center image onto a black background of given size 
            img = cv2.resize(img, dsize=None, fx = dcm.PixelSpacing[1]/self.pixel_spacing[1], fy=dcm.PixelSpacing[0]/self.pixel_spacing[0])
            img = utils.resize_dicom(img, self.output_size, self.background_HU)

        if self.window is not None:
            
            img = utils.apply_HU_window(img, self.window, dcm)
        
        if self.debug==True:
            return img0, keep_mask, img # image before cropping, cropping mask, image after processing
        
        return img, dcm # image after processing, pydicom object
        
    def crop(self, dcm, img):
        
        pixel_size_mm2 = np.prod(dcm.PixelSpacing) # actual pixel size on the image

        # remove small objects (patient table, arm, etc) 
        # min_body_area_mm2 is defined heuristically
        # mask_HU_thr is the lower edge of the HU window 
        keep_mask = skimage.morphology.remove_small_objects(img>self.mask_HU_thr, min_size=int(self.min_body_area_mm2/pixel_size_mm2))
        
        # remove holes in the body cross-section (lungs, etc)
        # max_hole_area_mm2 is defined heuristically 
        keep_mask = skimage.morphology.remove_small_holes(keep_mask, area_threshold=int(self.max_hole_area_mm2/pixel_size_mm2))
    
        img[~keep_mask] = self.background_HU # erase everything outside the body mask by assigning to background
    
        x, y, w, h = cv2.boundingRect((keep_mask*255).astype('uint8')) # obtain rectangular area around the mask
        
        return img[y:y+h, x:x+w], keep_mask