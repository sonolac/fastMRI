       
get_ipython().run_line_magic('matplotlib', 'inline')

import h5py
import numpy as np
from matplotlib import pyplot as plt
from data import transforms as T

def show_slices(data, slice_nums, cmap=None):
    fig = plt.figure()
    for i, num in enumerate(slice_nums):
        plt.subplot(1, len(slice_nums), i + 1)
        plt.imshow(data[num], cmap=cmap)

file = '../fastMRIData/singlecoil_val/file1000000.h5'
hf = h5py.File(file)



print('Keys:', list(hf.keys()))
print('Attrs:', dict(hf.attrs))


volume_kspace = hf['kspace'][()]
print(volume_kspace.dtype)
print(volume_kspace.shape)

slice_kspace = volume_kspace # Choosing the 20-th slice of this volume
show_slices(np.log(np.abs(slice_kspace) + 1e-9), [0, 5, 10])

# In[9]:


slice_kspace2 = T.to_tensor(slice_kspace)      # Convert from numpy array to pytorch tensor
slice_image = T.ifft2(slice_kspace2)           # Apply Inverse Fourier Transform to get the complex image
slice_image_abs = T.complex_abs(slice_image)   # Compute absolute value to get a real image


# In[10]:


show_slices(slice_image_abs, [0, 5, 10], cmap='gray')


# As we can see, each slice in a multi-coil MRI scan focusses on a different region of the image. These slices can be combined into the full image using the Root-Sum-of-Squares (RSS) transform.

# In[11]:


slice_image_rss = T.root_sum_of_squares(slice_image_abs, dim=0)


# In[12]:


plt.imshow(np.abs(slice_image_rss.numpy()), cmap='gray')


# So far, we have been looking at fully-sampled data. We can simulate under-sampled data by creating a mask and applying it to k-space.

# In[13]:


from common.subsample import MaskFunc
mask_func = MaskFunc(center_fractions=[0.04], accelerations=[8])  # Create the mask function object


# In[14]:


masked_kspace, mask = T.apply_mask(slice_kspace2, mask_func)   # Apply the mask to k-space


# Let's see what the subsampled image looks like:

# In[15]:


sampled_image = T.ifft2(masked_kspace)           # Apply Inverse Fourier Transform to get the complex image
sampled_image_abs = T.complex_abs(sampled_image)   # Compute absolute value to get a real image
sampled_image_rss = T.root_sum_of_squares(sampled_image_abs, dim=0)


# In[16]:


plt.imshow(np.abs(sampled_image_rss.numpy()), cmap='gray')


