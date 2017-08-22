'''
    File name         : compression.py
    File Description  : Image Compression Algorithms SVD,QR and LU
    Author            : Srini Ananthakrishnan
    Date created      : 10/16/2016
    Date last modified: 10/16/2016
    Python Version    : 3.5
'''

from common import *

#===================================#
# Sub-routine for SVD using QR Eigen
#===================================#

def qr_eigen(M, maxIter):
    """Function computes eigenvalue and eigenvector using QR householder reflection
    Args:
        M       : Input square matrix
        maxIter : Maximum Iterations to determine eigen
    Return:
        Eigenvalue, Eigenvector
    """
    # Initialize empty array to store eigenvalues
    A = []
    
    # Q is identity matrix of size M rows
    Q = np.eye(M.shape[0])
    
    # Append input matrix M to A
    A.append(None)
    A.append(M)
    
    # Loop for max-iteration and compute eigenvalue and eigenvector
    # using QR householder reflection
    for k in range(maxIter):
        A[0] = A[1]
        q, R = householder_reflection(A[0])
        A[1] = np.dot(R, q)
        Q = Q.dot(q)
        #print (k, "/", maxIter)
    return np.diagonal(A[1]), Q

#=========================================#
# Sub-routine's for SVD using power method
#=========================================#

def generateRandomUnitVector(n):
    """Function generates random UNIT vector (l2-normalized) in R^n vector space
    Args:
        n : dimention of vector space
    Return:
        The return value. Random unit vectors in in R^n vector space
    """
    # normalvariate method generates "n" random numbers that are
    # normally distributed with mean=0 and standard deviation=1
    vectors = [normalvariate(0, 1) for _ in range(n)]
    dprint("un-normalized random numbers (x):\n",vectors)
    
    # L2-Norm: Normalize random unit vector
    # ‖x‖ = sqrt(x_1^2 + x_2^2 +...+ x_n^2).
    l2Norm = sqrt(sum(x * x for x in vectors))
    dprint("l2Norm:\n",l2Norm)
    
    # Dividing a vector by its norm gives a unit vector :
    # For any scalar constant c, we know vector x has property ‖cx‖=|c|⋅‖x‖, 
    # where ‖x‖ is l2-norm of vector x. Given that unit vector can be 
    # generated when c = 1, then ‖(x/‖x‖)‖ = (1/‖x‖)(‖x‖) = 1
    unitVectors = [x / l2Norm for x in vectors]
    
    return unitVectors
 

def computeOneDimensionSVD(A, epsilon=1e-10):
    """Function computes single dimension SVD
    Args:
        A       : input square matrix
        epsilon : 
    Return:
        The return value. Singular unit vector
    """
    # get the size of rows and cols 
    n, m = A.shape
    
    # generate random unit vector x (l2-normalized) in R^n vector space
    x = generateRandomUnitVector(m)
    dprint("x,m:\n",x,m)
    
    # track currentVector and lastVector to be iterated
    lastVector = None
    currentVector = x
    dprint("currentVector:\n",currentVector)

    # construct a matrix B that will rotate and stretch the unit 
    # vector in R^n vector space    
    if n >= m:
        B = np.dot(A.T, A)
    else:
        B = np.dot(A, A.T)    
    dprint("B:\n",B)
    
    # iterate until dot product of current and last vector is just exceed value 1
    # and return the current vector ()
    # remember we are trying to determine orthonormal unit vector
    iterations = 0
    while True:
        iterations += 1
        lastVector = currentVector
        dprint("lastVector-inloop:\n",lastVector)
        
        # currentVector (x_t+1) = B.x_t
        currentVector = np.dot(B, lastVector)
        
        # re-normalize at each step
        currentVector = currentVector / norm(currentVector)
        dprint("currentVector-inloop:\n",currentVector)
        
        # magnitude of the dot product between currentVector x_t+1
        # and lastVector x_t is very close to 1 (say, 1-epsilon = 0.99999)
        if abs(np.dot(currentVector, lastVector)) > 1 - epsilon:
            dprint("currentVector:\n",currentVector)
            dprint("converged in {} iterations!".format(iterations))
            return currentVector


def svd(A, epsilon=1e-10):
    """Function computes SVD of matrix
    Args:
        A       : input square matrix
        epsilon : 
    Return:
        The return value. U, V and sigma
    """
    # get the size of rows and cols 
    n, m = A.shape
    
    # initialize array to store computed U,V and sigma
    svdComputed = []

    k = min(n,m) 
    # loop thru col vector range
    for i in range(k):
        # make a copy of input matrix
        matrixForOneDim = A.copy()
        dprint("matrixForOneDim:\n",matrixForOneDim)
        dprint("svdComputed[:i]:\n",svdComputed[:i])
        
        # subtract the rank 1 component of A corresponding to v_1. i.e., 
        # set A' = A - sigma_1(A) u_1 v_1^T. Then it’s easy to see that 
        # sigma_1(A') = sigma_2(A) and basically all the singular vectors 
        # shift indices by 1 when going from A to A'. Then repeat in loop
        for singularValue, u, v in svdComputed[:i]:
            dprint("singularValue:\n",singularValue)
            dprint("u:\n",u)
            dprint("v:\n",v)
            dprint("np.outer(u,v):\n",np.outer(u, v))
            matrixForOneDim = matrixForOneDim - (singularValue * np.outer(u, v))
            
        dprint("matrixForOneDim:\n",matrixForOneDim)
        # next sigular vector
        v = computeOneDimensionSVD(matrixForOneDim, epsilon=epsilon)  
        dprint("v after computeOneDimensionSVD:\n",v)
        
        # u vector unnormalized is dot product of A and sigular vector v
        uVectorUnnormalized = np.dot(A, v)
        dprint("uVectorUnnormalized:\n",uVectorUnnormalized)
        
        # compute sigma (next singular value) from normalized u vector
        sigma = norm(uVectorUnnormalized)
        dprint("sigma:\n",sigma)
        
        # compute u = u-vector unnormalized / sigma
        u = uVectorUnnormalized / sigma
        dprint("u:\n",u)
        
        # append for each iteration of col vector
        svdComputed.append((sigma, u, v))
        dprint("svdComputed:\n",svdComputed)
        
    # transform it into matrices of the right shape
    singularValues, us, vs = [np.array(x) for x in zip(*svdComputed)]
    dprint("singularValues:\n",singularValues)
    dprint("u:\n",us.T)
    dprint("v:\n",vs)
    # return U,sigma and V
    return us.T, singularValues, vs

def svd_power_plot(U,V,sigma,imgmat):
    ssim_val = []
    rank = []
    mse_val = []
    for i in range(5, 50, 10):
        reconstimg = np.matrix(U[:, :i]) * np.diag(sigma[:i]) * np.matrix(V[:i, :])
        print("\t\t   rank = ",i)
        rank.append(i)
        ssim_val.append(ssim(imgmat, reconstimg))
        mse_val.append(mse(imgmat, reconstimg))
        compare_images(imgmat, reconstimg, "Original vs. Reconstructed");

#=========================================#
# Sub-routine's for QR decomposition
#=========================================#

def householder_reflection(A):
    """Perform QR decomposition of matrix A using Householder reflection."""
    (rows, cols) = np.shape(A)

    # Initialize orthogonal matrix Q and upper triangular matrix R.
    # Set Q to Identity Matrix and R to input matrix A
    Q = np.identity(rows)
    R = np.copy(A)

    # Loop over each column subvector
    for i in range(rows - 1):
        # Store the subvector in x
        x = R[i:, i]
        dprint("i ", i, "x", x)        
        
        e = np.zeros_like(x)
        # Compute norm and copy sign of the subvector
        e[0] = copysign(np.linalg.norm(x), -A[i, i])
        dprint("e: ", e)

        # Build u from the subvector and the norm
        u = x + e 
        dprint("u:", u)

        # Compute norm of u and store as v
        v = u / np.linalg.norm(u)
        dprint("v:", v)

        # Build Householder reflection
        Q_i = np.identity(rows)
        Q_i[i:, i:] -= 2.0 * np.outer(v, v)        
        dprint("Q[",i,"]")
        dprint(Q_i)

        # Apply this householder reflection to R, and then to Q
        R = np.dot(Q_i, R)
        Q = np.dot(Q, Q_i.T)
        
    # Return Q and R
    return (Q, R)
    
#=========================================#
# Sub-routine's for LU decomposition
#=========================================#

def mult_matrix(M, N):
    """Multiply square matrices of same dimension M and N"""
    C = (M.T.dot(N))
    #print("Pivoted matrix: ")
    #print(C)
    return C

def pivot_matrix(M):
    """Returns the pivoting matrix P for M, as used in Doolittle's method."""
    m = len(M)
    # *Create an identity matrix, with floating point values. You may use numpy
    id_mat = [[float(i ==j) for i in range(m)] for j in range(m)]

    row = 0
    
    # Rearrange the identity matrix such that the largest element of                                                                                                                                                                                   
    # each column of M is placed on the diagonal of of M
    # for every row in the input matrix
    for j in range(m):
        # *find the row with the biggest element in column j (so we are looking for the diagonal elements M[j,j])
        # *we do not want to be swapping with rows above this one, just the rows below it.
        #row = np.argmax(M, axis=0)[j]
        row = max(range(j, m), key=lambda i: abs(M[i][j]))        
        if j != row: #if this row is not the row with the next biggest diagonal element
            #pass
            # *Swap the rows of the id matrix 
            id_mat[j], id_mat[row] = id_mat[row], id_mat[j]

    # Return id_mat as numpy array
    id_mat = np.asarray(id_mat)
    #print("Pivot: ")
    #print(id_mat)
    return id_mat

def LU_decompose(PA, L, U, n):
    """Performs the actual LU decomposition using the standard formula"""
    for j in range(n):
        
        # All diagonal entries of L are first set to 1, you may use numpy to do this as well.                                                                                                                                                                                                
        L[j][j] = 1.0

        # *Encode the following logic:
        # *LaTeX: $u_{ij} = a_{ij} - \sum_{k=1}^{i-1} u_{kj} l_{ik}$                                                                                                                                                                                     
        for i in range(j+1):
            eqU = sum(U[k][j] * L[i][k] for k in range(i))
            U[i][j] = PA[i][j] - eqU

        # *Encode the following logic:
        # *LaTeX: $l_{ij} = \frac{1}{u_{jj}} (a_{ij} - \sum_{k=1}^{j-1} u_{kj} l_{ik} )$                                                                                                                                                                  
        for i in range(j, n):
            eqL = sum(U[k][j] * L[i][k] for k in range(j))
            L[i][j] = (PA[i][j] - eqL) / U[j][j]
            
    return L, U

def lu_decomposition(A):
    """Performs an LU Decomposition of A (which must be square)                                                                                                                                                                                        
    into PA = LU. The function returns P, L and U."""
    n = len(A)

    # Create zero matrices for L and U or use np.zeros(())                                                                                                                                                                                                                
    L = np.asarray([[0.0] * n for i in range(n)])
    U = np.asarray([[0.0] * n for i in range(n)])
    
    # create the pivot matrix P and the matrix product PA                                                                                                                                                                                            
    P = pivot_matrix(A)
    PA = mult_matrix(P, A)
    
    # Decompose
    L, U = LU_decompose(PA, L, U, n)
        
    return (P, L, U)

#======================================#
# Section 1 Algorithms Python Routine's 
#======================================#

#==================================================#
# Sigular Value Decomposition (SVD)                #
# Section 1.1 Algorithm 1 - SVD using Power Method #
#==================================================#

def svd_power(imgmat):
    """Function computes SVD using power method
    Args:
	None
    Return:
        None
    """

    A = imgmat
    A = np.asarray(A)

    U, sigma, V = svd(A)

    dprint("U:\n",U)
    dprint("sigma:\n",sigma)
    dprint("V:\n",V)

    # Diaganalose singular (S) matrix
    #S = np.diag(sigma)
    m = U.shape[0]
    n = V.shape[0]
    s = sigma.shape[0]
    dprint(U.shape,V.shape,sigma.shape)
    S = np.zeros((n, n))
    S[:s, :s] = np.diag(sigma)


    reconimage = np.dot(U, np.dot(S, V))
    
    # plot the reconstructed image
    #title = "SVD power method decomposition"
    #recon_img_plot(title,reconimage)

    # compare the images
    print("(Original == Reconstructed) : ",np.allclose(A, reconimage))
    compare_images(A, reconimage, "Original vs. Reconstructed");
    
    # Plot rank based image construction
    svd_rank_plot(U,V,sigma,imgmat)

#=====================================================================#
# Section 1.2 Algorithm 2 - SVD using QR Eigen (Householder Reflection) 
#=====================================================================#

def svd_qr(imgmat):
    """Function computes SVD using QR determined eigenvalues and eigenvector
    Args:
	None
    Return:
        None
    """

    # Copy image matrix as np.array for computation ease.
    A = np.asarray(imgmat)
    dprint(A.shape)
    m,n = A.shape
    dprint(m,n)
    
    # Get transpose matrix
    AAt = A.dot(A.T)
    AtA = A.T.dot(A)

    # Get the Eigenvalues and Eigenvectors
    Su, U = qr_eigen(AAt,6)
    Sv, V = qr_eigen(AtA,6)

    dprint(U.shape,V.shape,Su.shape,Sv.shape)
    # Compute singular values (S1) from square root of Eigenvalues (Su)
    # Sort singular values in descending order and store them in order
    with np.errstate(invalid='ignore'):
        Sv = np.sqrt(Sv);
    idxv = Sv.argsort()[::-1]
    with np.errstate(invalid='ignore'):
        Su = np.sqrt(Su);
    idxu = Su.argsort()[::-1]

    if m > n:
        sigma = Sv[idxv]
    else:
        sigma = Su[idxu]
        
    U = U[:,idxu]
    V = V[:,idxv]
            
    # Transpose row (rotation) matrix V    
    Vt = V.T

    # Diaganalose singular (S) matrix
    s = sigma.shape[0]
    S = np.zeros((m, n))
    S[:s, :s] = np.diag(sigma)

    dprint(U.shape,Vt.shape,S.shape,sigma.shape)
    # Reconstruct imgae with dot product of ALL 3 matrices
    reconimage = np.dot(U, np.dot(S, Vt))

    # plot the reconstructed image
    #title = "SVD QR method decomposition"
    #recon_img_plot(title,reconimage)
    
    # compare the images
    dprint("(Original == Reconstructed) : ",np.allclose(A, reconimage))
    compare_images(imgmat, reconimage, "Original vs. Reconstructed");

    # Plot rank based image construction
    #svd_rank_plot(U,V,sigma,imgmat)

#============================================#
# QR Decomposition                           #
# Section 1.3 Algorithm 3 - QR decomposition # 
#============================================#

def qr_householder(imgmat):
    """Function computes QR decomposition using householder reflection 
       and image reconstruction
    Args:
	None
    Return:
        None
    """
    # make a local copy of image
    A = imgmat
    A = np.asarray(A)
    if (A.shape[0] != A.shape[1]):
        print("Only supported for m x m sqaure image matrix")
        return

    (Q,R) = householder_reflection(A)
    reconimage = Q.dot(R)

    #title = "Householder QR decomposition"
    #recon_img_plot(title,reconimage)
    
    # compare the images
    print("(Original == Reconstructed) : ",np.allclose(A, reconimage))
    compare_images(A, reconimage, "Original vs. Reconstructed");
    


#============================================#
# LU Decomposition                           #
# Section 1.4 Algorithm 4 - LU decomposition #
#============================================#

def lu(imgmat):
    """Function computes LU decomposition and image reconstruction
    Args:
	None
    Return:
        None
    """
    A = imgmat
    A = np.asarray(A)
    
    if (A.shape[0] != A.shape[1]):
        print("Only supported for m x m sqaure image matrix")
        return

    P, L, U = lu_decomposition(A)
    reconimage = np.array(P).dot(L).dot(U)
    
    title = "LU decomposition"
    recon_img_plot(title,reconimage)

    # compare the images
    compare_images(A, reconimage, "Original vs. Reconstructed");
