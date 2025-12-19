import cv2
import numpy as np

def apply_filter(image, filter_name):
    """
    Apply selected filter to image
    Returns: Filtered image
    """
    if filter_name == 'normal':
        return image
    
    elif filter_name == 'bw':
        # Convert to grayscale (Black & White)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # Convert back to 3 channels for consistent output
        return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    
    elif filter_name == 'vintage':
        # Create vintage sepia effect
        height, width = image.shape[:2]
        
        # Create sepia matrix
        sepia_filter = np.array([
            [0.393, 0.769, 0.189],
            [0.349, 0.686, 0.168],
            [0.272, 0.534, 0.131]
        ])
        
        # Apply sepia filter
        sepia = cv2.transform(image, sepia_filter)
        
        # Clip values to valid range
        sepia = np.clip(sepia, 0, 255)
        
        # Add slight vignette effect
        kernel_x = cv2.getGaussianKernel(width, width/3)
        kernel_y = cv2.getGaussianKernel(height, height/3)
        kernel = kernel_y * kernel_x.T
        
        mask = kernel / kernel.max()
        vignette = np.ones_like(sepia, dtype=np.float32)
        vignette[:,:,0] *= mask
        vignette[:,:,1] *= mask
        vignette[:,:,2] *= mask
        
        result = sepia * vignette
        result = np.clip(result, 0, 255).astype(np.uint8)
        
        # Add subtle noise for film grain effect
        noise = np.random.normal(0, 3, result.shape).astype(np.int16)
        result = np.clip(result.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        
        return result
    
    else:
        return image