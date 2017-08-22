'''
    File name         : common.py
    File Description  : File contains common plot and debug methods
    Author            : Srini Ananthakrishnan
    Date created      : 10/16/2016
    Date last modified: 10/16/2016
    Python Version    : 3.5
'''

# Import Python Libraries
from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.cm as cm
import time
import scipy
import scipy.linalg

from sklearn.metrics import mean_squared_error
from skimage.measure import compare_ssim as ssim
from PIL import Image
from numpy.linalg import norm
from random import normalvariate
from math import sqrt
from math import copysign

# open image file and store as numpy matrix
# img = Image.open('baboon-grayscale.png')
# imggray = img.convert('LA')

# imgmat = np.array(list(imggray.getdata(band=0)), float)
# imgmat.shape = (imggray.size[1], imggray.size[0])
# imgmat = np.matrix(imgmat)


def dprint(*args, **kwargs):
    """Debug print function using inbuilt print
    Args:
        args   : variable number of arguments
        kwargs : variable number of keyword argument
    Return:
        None. 
    """
    #print(*args, **kwargs)
    pass

def orig_img_plot(imgmat):
    """Plot original image
    Args:
        None
    Return:
        None. 
    """
    print(imgmat.shape)
    title = "Original Image"
    plt.title(title)
    plt.imshow(imgmat, cmap='gray');

def recon_img_plot(title,reconimage):
    """Plot reconstructed image
    Args:
        None
    Return:
        None. 
    """
    plt.title(title)
    plt.imshow(reconimage, cmap='gray');

def mse(imageA, imageB):
    """Calculate 'Mean Squared Error'(MSE) between the two images
    Args:
        imageA : input image A
        imageB : input image B
    Return:
        The return value is error (MSE)
    """
    # the 'Mean Squared Error' between the two images is the
    # sum of the squared difference between the two images;
    # NOTE: the two images must have the same dimension
    err = np.sum((imageA.astype("float") - imageB.astype("float")) ** 2)
    err /= float(imageA.shape[0] * imageA.shape[1])
    
    # return the MSE, the lower the error, the more "similar"
    # the two images are
    return err
 
def compare_images(imageA, imageB, title):
    """Plot and compare two images MSE and SSIM
    Args:
        imageA : input image A
        imageB : input image B
        title  : title on the plot
    Return:
        None
    """
    #print("---------------------------------")
    #print("Plot Original vs. Reconstructed:")
    #print("---------------------------------")

    # compute the mean squared error and structural similarity
    # index for the images
    if (imageA.shape[0] == imageA.shape[1]):
        m = mse(imageA, imageB)
        s = ssim(imageA, imageB)
    else:
        m = 0
        s = 1 # scikit learn ssim has issues with non-square images
    # setup the figure
    fig = plt.figure(title)
    plt.suptitle("MSE: %.2f,  SSIM: %.2f" % (m, s),fontsize=12)
 
    # show first image
    ax = fig.add_subplot(1, 2, 1)
    plt.title("Original")
    plt.imshow(imageA, cmap = plt.cm.gray)
    #plt.axis("off")
 
    # show the second image
    ax = fig.add_subplot(1, 2, 2)
    plt.title("Reconstructed")
    plt.imshow(imageB, cmap = plt.cm.gray)
    #plt.axis("off")
 
    # show the images
    plt.show()


def svd_graph(ssim_val,rank,mse_val):
    """Plot SVD graphs MSE and SSIM vs Singular values
    Args:
        imageA : input image A
        imageB : input image B
        title  : title on the plot
    Return:
        None
    """
    # setup the figure
    fig = plt.figure()
    
    # setup subplots for MSE and SSIM
    ax1 = fig.add_subplot(211)
    ax2 = fig.add_subplot(212)

    
    # plot singular values vs ssim
    ax1.plot(rank,ssim_val)
    
    print("-------------------------------")
    print(" MSE/SSIM vs Singular Values:")
    print("-------------------------------")
    
    # plot singular values vs mse
    ax2.plot(rank,mse_val)
    
    # set x-axis and y-axis labels
    ax1.set_ylabel('SSIM')
    ax1.set_xlabel('Number of Singular Values')
    ax1.set_title('Figure: SSIM vs Singular Values')
    ax2.set_ylabel('MSE')
    ax2.set_xlabel('Number of Singular Values')
    ax2.set_title('Figure: MSE vs Singular Values')
    
    fig.tight_layout()

    # show the plot
    plt.show()

def svd_rank_plot(U,V,sigma,imgmat):
    """Plot SVD image by rank
    Args:
        imageA : input image A
        imageB : input image B
        title  : title on the plot
    Return:
        None
    """
    print("---------------------------------")
    print("Plot reconstructed image by rank:")
    print("---------------------------------")
    m = U.shape[0]
    n = V.shape[0]
    ssim_val = []
    rank = []
    mse_val = []
    for i in range(5, 50, 10):
    	reconstimg = np.matrix(U[:, :i]) * np.diag(sigma[:i]) * np.matrix(V[:i, :])
    	print("\t\t   rank = ",i)
    	rank.append(i)
    	# scikit learn ssim has issues with non-square images
    	if (m == n): 
    	    ssim_val.append(ssim(imgmat, reconstimg))
    	    mse_val.append(mse(imgmat, reconstimg))
            
    	compare_images(imgmat, reconstimg, "Original vs. Reconstructed");
    if (m == n):
        svd_graph(ssim_val,rank,mse_val)

  
def main():
    pass


if __name__ == "__main__":
    main()

