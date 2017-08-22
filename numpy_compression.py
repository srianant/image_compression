'''
    File name         : numpy_compression.py
    File Description  : Standard Numpy/Scipy Imgae Compression Methods
    Author            : Srini Ananthakrishnan
    Date created      : 10/16/2016
    Date last modified: 10/16/2016
    Python Version    : 3.5
'''

# import user defined common methods
from common import *

#======================================#
# Section 2 Using Standard Library 
#======================================#

#======================#
# 2.1 SVD using Numpy  #
#======================#

def numpy_svd(imgmat):
    """Function computes numpy based SVD and reconstruct image 
    Args:
        None
    Return:
        None
    """
    
    # make a local copy of image
    A = imgmat
    dprint(imgmat.shape)
    A = np.asarray(A)    
    # compute SVD using numpy library
    U, sigma, V = np.linalg.svd(A)
    
    dprint(U.shape,V.shape,sigma.shape)
    
    m = U.shape[0]
    n = V.shape[0]
    s = sigma.shape[0]
    dprint(U.shape,V.shape,sigma.shape)
    # diagonalize singular values
    #S = np.diag(sigma)
    #S = np.zeros((m, n), dtype=complex)
    S = np.zeros((m, n))
    S[:s, :s] = np.diag(sigma)
    
    # reconstruct image
    reconimage = np.dot(U, np.dot(S, V))
    dprint(reconimage.shape)
    
    # plot the reconstructed image
    title = "Numpy SVD reconstructed image"
    recon_img_plot(title,reconimage)
    
    # compare the images
    print("(Original == Reconstructed) : ",np.allclose(A, reconimage))
    #compare_images(A, reconimage, "Original vs. Reconstructed");
    
    # plot rank wise image construction
    svd_rank_plot(U,V,sigma,imgmat)

#======================#
# 2.2 QR using Numpy   #
#======================#

def numpy_qr(imgmat):
    """Function computes numpy based QR decomposition and reconstruct image 
    Args:
        None
    Return:
        None
    """

    # make a local copy of image
    A = imgmat
    
    # compute QR using numpy library
    Q, R = np.linalg.qr(A)

    # reconstruct image
    reconimage = np.dot(Q,R)

    # plot the reconstructed image
    #title = "Numpy QR reconstructed image"
    #recon_img_plot(title,reconimage)
    
    # compare the images
    print("(Original == Reconstructed) : ",np.allclose(A, reconimage))
    compare_images(A, reconimage, "Original vs. Reconstructed");

#======================#
# 2.3 LU using Scipy   #
#======================#

def scipy_lu(imgmat):
    """Function computes scipy based LU decomposition and reconstruct image 
    Args:
        None
    Return:
        None
    """

    # make a local copy of image
    A = imgmat

    # compute LU using scipy library
    P, L, U = scipy.linalg.lu(A)
    dprint(P.shape,L.shape,U.shape)
    # reconstruct image
    reconimage = np.array(P).dot(L).dot(U)

    # plot the reconstructed image
    #title = "Scipy LU reconstructed image"
    #recon_img_plot(title,reconimage)
    
    # compare the images
    print("(Original == Reconstructed) : ",np.allclose(A, reconimage))
    compare_images(A, reconimage, "Original vs. Reconstructed");




