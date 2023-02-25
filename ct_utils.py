import PIL.Image as pil_image
import io
import numpy as np
import matplotlib.pyplot as plt
import numpy as np

def ctload(image_path):
    with open(image_path, 'rb') as f:
        tif = pil_image.open(io.BytesIO(f.read()))
    return np.array(tif)


def ctshow(image_arr):
    plt.close()
    plt.imshow(image_arr, cmap='gray')#, vmin=0, vmax=1)
    plt.show()
    
def ctsave(image_arr, out_path):
    plt.close()
    plt.imshow(image_arr, cmap='gray', dpi=300)
    plt.savefig(out_path, dpi=300)    

