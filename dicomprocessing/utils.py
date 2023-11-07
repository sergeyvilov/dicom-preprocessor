import numpy as np
import png
    
def resize_dicom(img, output_size, background_HU=-1000):
                        
    height_new, width_new = output_size # required dimensions
        
    height, width = img.shape # actual dimensions

    img = img[max((height-height_new)//2,0):(height+height_new)//2,
             max((width-width_new)//2,0):(width+width_new)//2] # use the central part of the image if actual dimensions are larger than required dimensions
        
    height, width = img.shape # recompute dimensions after cropping the central part

    # position of the old image inside the new image
    x0 = width_new//2 - width//2 
    y0 = height_new//2 - height//2
                
    img_new = np.ones((height_new, width_new))*background_HU # fill new image with background (air)
        
    img_new[y0:y0 + height, x0:x0 + width] = img # place the old image in the center of the new image
        
    return img_new
    
def apply_HU_window(img, window, dcm=None):
        
    if window == 'dicom':

        WindowCenter, WindowWidth = dcm.WindowCenter, dcm.WindowWidth

    else:

        WindowCenter, WindowWidth = window[0], window[1]

    vmin = WindowCenter - WindowWidth/2
    vmax = WindowCenter + WindowWidth/2
        
    img = np.clip(img, vmin, vmax) # clip
        
    img = (img - vmin)/(vmax - vmin) # stretch, we don't apply minmax to the image! we use window edges to preserve infor about HU units 
                
    return img

def save_png(output_path, img, WindowWidth):
    '''
    Rescale the image to WindowWidth and save as .png
    '''

    bitdepth = np.ceil(np.log2(WindowWidth)).astype(int)

    with open(output_path, 'wb') as f:
        png_writer = png.Writer(img.shape[1], img.shape[0], bitdepth=bitdepth)  
        png_writer.write(f, (img*WindowWidth).astype(int))