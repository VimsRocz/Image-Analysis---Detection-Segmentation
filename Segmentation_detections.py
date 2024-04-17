#!/usr/bin/env python
# coding: utf-8

# 

# # Image analysis 
# ##  Detection / Segmentation

# | Group __25__ | Name  | Matr.-nr. |
# |-|-|-|
# |   Member 1  | Vimal Chawda | 10025862 |
# |   Member 2  |
# |   Member 3  | name3 | 12345 |

# Required __imports__ for this lab. Don't forget to run the following cell (which may __take up some minutes__ because some functions are getting compiled).

# In[2]:


import lab                       # Given functions
from numba import jit            # Faster computations
import numpy as np               # Numerical computations
import math 


# ## Exercise 1: Point detection
# ### Exercise 1.1: Scale adapted Harris- and Foerstner-Operator
# __Complete__ the function in the next cell that applies the Harris-Operator and the Foerstner-Operator for corner detection. The parameters are:
# - Image $I$, in which corners should be detected
# - Differentiation scale $\sigma_\Delta$
# - Factor $k$ between the integration scale $\sigma_{I}$ and differentiation scale: $\sigma_{I} = \dfrac{\sigma_\Delta}{k}$
# - Minimum isotropy $q_{min}$ for point regions (Foerstner-Operator)
# - Cornerness parameter $\kappa$ (Harris-Operator)
# - Number of best points to draw

# In[2]:


@jit(nopython=True, cache=True) # Uncomment to improve computation speed
def scale_adapted_harris_foerstner(I, sigma_dif, k, q_min, kappa, num_points):
    h, w, d = I.shape
    assert d == 1, "Only valid for single channel images!"
    
    Ix, Iy = lab.gaussian_derivatives(I, sigma_dif)  # Compute gaussian derivatives
    Ixx = Ix * Ix
    Iyy = Iy * Iy
    Ixy = Ix * Iy
    
    SMP = np.dstack((Ixx, Ixy, Iyy)) # Squares and mixed products as 3D array with 3 channels (Ixx, Ixy, Iyy)
    
    sigma_int    = sigma_dif / k                     # Compute the integration scale
    SMP_smoothed = lab.gaussian_blur(SMP, sigma_int) # Smoothing of squares and mixed products
        
    
    omega_map = np.zeros((h,w), dtype=np.float64)    # 2D array, holding the point weight for each pixel
    q_map = np.zeros((h,w), dtype=np.float64)        # 2D array, holding the isotropy for each pixel
    r_map  = np.zeros((h,w), dtype=np.float64)       # 2D array, holding the cornerness values for each pixel
    
    candidates_harris        = [(0, 0)][1:]          # Point regions as list of tuples (x, y)
    candidates_foerstner     = [(0, 0)][1:]          
    candidates_harris_max    = [(0, 0)][1:]          # Local maximas as list of tuples (x, y)
    candidates_foerstner_max = [(0, 0)][1:]          
    
    I_harris_corners     = np.dstack((I,I,I))        # 3-channel copies of the image (to draw points)
    I_foerstner_corners  = np.dstack((I,I,I))    
    
    M = np.zeros((2, 2), dtype=np.float64)           # 2 x 2 Matrix M - autocorrelation matrix
  

    for x in range(h):
        for y in range(w):
            
            # YOUR CODE GOES HERE
            #
            # Step 1: assign the values of SMP_smoothed at (x,y) to M
            

            M[0,0] = SMP_smoothed[x,y,0]
            M[0,1] = SMP_smoothed[x,y,1] 
            M[1,0] = M[0,1]
            M[1,1] = SMP_smoothed[x,y,2]
            # Step 2: compute the determinant and trace of M
            
            det = np.linalg.det(M)
            trace = np.trace(M)
            
            # Step 3: compute the point weight 'omega' and the isotropy 'q'
            
            omega = det/trace; 
            q=4*(det/(trace**2))   #trace^2
            
            # Step 4: compute 'r', the cornerness
            
            r=det - kappa*trace*trace
            
            # Hints:  you can use the functions:
            #         - np.linalg.det
            #         - np.trace

            omega_map[x, y] = omega
            q_map[x, y]     = q
            r_map[x, y]     = r

            candidates_harris.append((x, y))
            if q > q_min:
                candidates_foerstner.append((x, y))
                
    # Non-max supression (5x5 window)
    
    for cx, cy in candidates_harris:  
        r = r_map[cx, cy]
        neighbours = lab.get_valid_neighbours_dist(h, w, cx, cy, 2)       
        is_r_max = True
        for nx, ny in neighbours:
            if r_map[nx, ny] >= r:
                is_r_max = False
                break
        if is_r_max:
            candidates_harris_max.append((cx, cy))
                 
    for cx, cy in candidates_foerstner:    
        omega = omega_map[cx, cy]
        neighbours = lab.get_valid_neighbours_dist(h, w, cx, cy, 2)
        is_omega_max = True
        for nx, ny in neighbours:
            if omega_map[nx, ny] >= omega:
                is_omega_max = False
                break
        if is_omega_max:
            candidates_foerstner_max.append((cx, cy))
                
    assert len(candidates_harris_max)    >= num_points, "not enough harris-points detected"
    assert len(candidates_foerstner_max) >= num_points, "not enough foerstner-points detected"
    
    # Draw n best points
    
    for n in range(num_points):
        best_r = -1.0
        best_p = (0, 0)
        for cx, cy in candidates_harris_max:
            if r_map[cx, cy] > best_r:
                best_r     = r_map[cx, cy]
                best_p     = (cx, cy)
        candidates_harris_max.remove(best_p)
        neighbours = lab.get_valid_neighbours_dist(h, w, best_p[0], best_p[1], 2)
        for nx, ny in neighbours:
            I_harris_corners[nx, ny, :] = (255,0,0) 
        
        best_omega = -1.0
        best_p     = (0, 0)
        for cx, cy in candidates_foerstner_max:
            if omega_map[cx, cy] > best_omega:
                best_omega = omega_map[cx, cy]
                best_p     = (cx, cy)
        candidates_foerstner_max.remove(best_p)
        neighbours = lab.get_valid_neighbours_dist(h, w, best_p[0], best_p[1], 2)
        for nx, ny in neighbours:
            I_foerstner_corners[nx, ny, :] = (255,0,0) 
            
    return I_harris_corners, I_foerstner_corners


# DONE<font color='red'><b>[bug in computation of r (should be trace*trace)]</b></font>

# ### Exercise 1.2: Evaluation
# The code in the following cell will evaluate the implementation. __Experiment with the differentiation scale and the minimum isotropy__ and __write__ a brief discussion, which contains the following aspects:
# 
# - Meaning of $M$ w.r.t. the detection of corners.
# - Similarities and differences between the Harris- and Foerstner-Operator (w.r.t. the procedure and the results).
# - Influence of $\sigma_\Delta$ on the result (with fixed $k$).
# - Meaning of $q_{min}$ and influence on the result.

# In[16]:


I = lab.imread3D('images/aerial_wide.jpg') # Load example image

# PARAMETERS:
sigma_dif = 2.99      # Integration scale [>=1]
k         = 0.6    # Factor between differentiation and integration scale [0.5 - 0.75]
kappa     = 0.04   # Parameter for the computation of the cornerness
q_min     = 0.49    # Threshold for the isotropy [0.0 - 1.0]
N         = 100    # Number of best points to draw

I_H, I_F = scale_adapted_harris_foerstner(I, sigma_dif, k, q_min, kappa, N)      

lab.imshow3D(I)
print('original image')
lab.imshow3D(I_H, I_F)
print('left to right: harris points | foerstner points')


# ## Discussion:
# 
# #### Meaning of $M$ w.r.t. the detection of corners.
# - Eigenvalues of Autocorrelation matrix contain information about the local image structure
# - Eigenvectors of M correspond to the main directions of the local image structure
# - Eigenvalues l1, l2 of M describe the average contrast in these directions
# A point of interest in image location can easily find in the image. As per lecture notes it is indepedent of the geometric transformation, scale variations, affine is also one form of transformation, illumination (pixel value remains constant)etc
# Features projection shows orientation if we know the orientation then we can find the location so if location is known then wrt every pixels we can scale variations.
# Information is mostly obtained near to the interest point so it is useful for processing.
# The points can be boundary corner points, corss , just small plots.
# #### Similarities and differences between the Harris- and Foerstner-Operator (w.r.t. the procedure and the results).
# 
# Harris corner detection algorithm is realized by calculating each pixel’s gradient (interesting points). If the absolute gradient values in two directions are both great, then judge the pixel as a corner.
# M is the autocorrelation matrix and it is also known as 2nd moments of gradient.
# The Harris corner detector is invariant to translation,rotation and illumination change. This detector is most repetitive and most informative. The disadvantage of this detector is it is not invariant to large scale change 
# - Harris and Foerstner use the same matrix but the calculation is different.
# Hence the autocorrelation matrix is different in Harris and Foerstner. 
# Both are applying local analysis of intensity function with the help of M autocorrelation Matrix.
# Harris operator - determinant + tracing to the Autocorrelation Matrix M to obtain the corner.
# Select the interest point as local maxima.
# Foestner operator - Compute the inverse the autocorrelation M matrix.
# Now we need to obtain interest points which will be obtain by eigen value. 
# Eigen value is the obtain from the inverse matrix is also know as error matrix as we can do some analysis as the axis represents Eigen values of inverse matrix. Error ellipse is circular and small with an isotropy range is from 0 to 1. 
# 
# #### Influence of $\sigma_\Delta$ on the result (with fixed $k$).
# 
# Differentiation scale and integration scale are correlated by the factor of k. 
# Sigma Integration Scale = k * Sigma Differentiation Scale
# If the value of K is constant then 
# both scale going to be larger so it will benifits in the process of better smoothness.
# Interesting points will be less in respective case.
# 
# <font color='red'><b>[Here k (NOT KAPPA) is the number of points to detect!]</b></font>
# 
# #### Meaning of $q_{min}$ and influence on the result.
# 
# q_min is the corner or end or last point in Foerstner operator. Hence as per lecture notes it is threshold used for the computation. 
# Isotropy 'q' = Eigen Values of the CV covariance matrix whose range is from 0 to 1. 
# As mention if error ellipse is circular then q = 1 (nearly to 1.)
# Set the q_min as our threshold, if the q_min increases then number of selected interesting points will be reduced as the detection of interesting point become more difficult or stonger in bound.

# ## Exercise 2: Edge pixel detection
# ### Exercise 2.1: Non max suppression w.r.t. the gradient directions
# __Complete__ the function below that performs a non max suppression of the gradient magnitudes with respect to the gradient directions. The parameters are: 
# 
# - $M$: __2D__ array of gradient magnitudes
# - $D$: __2D__ array of gradient directions in radians 
# 
# The function returns $R$, a __2D__ array where $R(x,y) = M(x,y)$ if $M(x,y)$ is a local maximum (w.r.t. the gradient direction) and $R(x,y) = 0$ else. 
# 
# The relevant neighbours w.r.t. the gradient angle $\Theta$ are shown in the following graphic.
# 
# <img src="./images/nonmaxdirs.png" width=75%>

# In[4]:


# @jit(nopython=True, cache=True) # Uncomment to improve computation speed
def non_max_sup_wrt_gradients(M, D):
    h, w = M.shape
    R    = np.zeros_like(M)     # Initialize result R with zeros (shape and type of I)
    
    for x in range(1, h - 1):
        for y in range(1, w - 1):
            magnitude = M[x, y]               # Gradient magnitude at (x,y)
            theta     = D[x, y] * 180 / np.pi # Gradient direction at (x,y) in degree
            
            # YOUR CODE GOES HERE
            #
            # Step 1: determine the coordinates of the two relevant neighbours (w.r.t. theta)
            if (theta<=22.5 and  theta>=-22.5) or (theta<=-157.5 or theta>=157.5):
                q = M[x-1,y]
                r = M[x+1,y]
                if q>magnitude or r>magnitude:
                    R[x,y]=0
                else:
                    R[x,y]=magnitude
            elif (theta<=67.5 and  theta>=22.5) or (theta<=-112.5 and theta>=-157.5):
                q = M[x-1,y-1]
                r = M[x+1,y+1]
                if q>magnitude or r>magnitude:
                    R[x,y]=0
                else:
                    R[x,y]=magnitude
            elif  (theta<=112.5 and  theta>=67.5) or (theta<=-67.5 and theta>=-112.5):
                q = M[x,y-1]
                r = M[x,y+1]
 # Step 2: check if the magnitude at (x,y) is bigger than the magnitude at the position of the two neighbours
                if q>magnitude or r>magnitude:
                    R[x,y]=0
                else:
                    R[x,y]=magnitude 
            elif  (theta<=157.5 and  theta>=112.5) or (theta<=-22.5 and theta>=-67.5):
                q = M[x+1,y-1]
                r = M[x-1,y+1]
# Step 3: if the criterion from Step 2 is fulfilled, set 'R[x, y]' to 'magnitude'
                if q>magnitude or r>magnitude:
                    R[x,y]=0
                else:
                    R[x,y]=magnitude 
    return R


# <font color='green' size='5'><b>OK</b></font>
# 
# <font color='orange'><b></b></font>

# ### Exercise 2.2: Hysteresis
# __Complete__ the function below, that performs hysteresis to determine whether a pixel is an edge pixel or not. The parameters are:
# 
# - $M_{max}$: __2D__ array of gradient magnitudes. Values of pixels which are not a local maximum are set to zero.
# - $m_{strt}$: Minimum gradient magnitude to start an edge
# - $m_{cont}$: Minimum gradient magnitude to continue an edge

# In[5]:


# @jit(nopython=True, cache=True) # Uncomment to improve computation speed
def apply_hysteresis(M_max, m_strt, m_cont):
    h, w = M_max.shape
    E    = np.zeros((h, w), dtype=np.bool_) # 3D-array of bools: true, if pixel is an edge-pixel
    
    candidates = [(0,0)][1:]        # Possible candidates for edge pixels as list of tuples (x,y). 
    for x in range(h):
        for y in range(w):
            if M_max[x,y] > m_strt:
                candidates += [(x, y)]  
         
    while len(candidates) > 0:
        (cx, cy) = candidates.pop() # Get candidate coordinates and remove it from the list 
        
        # YOUR CODE GOES HERE
        #
        # Step 1: skip candidate if (cx, cy) is already an edge pixel (check E at (cx, cy))
        if E[cx,cy] == True:
            continue
        elif M_max[cx,cy]>m_cont:
            E[cx,cy] = True
            neighbors = lab.get_valid_neighbours(h,w,cx,cy)
            for nx,ny in neighbors:
                candidates +=[(nx,ny)]

        # Step 2: if the gradient magnitude at (cx, cy) is bigger than m_cont:
        #         - set E at (cx, cy) to true
        #         - ad the 8 direct neighbours to the candidate list
        #           (you can use lab.get_valid_neighbours() for this)
            
    return E.reshape((h, w, 1)).astype(np.int64)*255 # modify E so it can be seen as single channel image


# DONE<font color='red'><b>[bug in this code.. candidates.pop() should not be in the if condidition. There 'continue' has to be used!]</b></font>

# #### Detector function
# The following function performs the actual detection using the functions above. No need to modify this, but __run the cell__! 

# In[6]:


def canny_edge_detector(I, sigma, m_strt, m_cont):
    assert I.shape[2] == 1,    "Expecting single channel image!"
    assert m_strt >= m_cont,   "Expecting m_strt >= m_cont!"
    assert m_cont >= 0.0,      "Expecting m_cont >= 0.0"

    M, D  = lab.gaussian_gradients(I, sigma)  
    M_max = non_max_sup_wrt_gradients(M[:,:,0], D[:,:,0])
    E     = apply_hysteresis(M_max, m_strt, m_cont)

    return E


# ### Exercise 2.3: Evaluation
# The following cell will apply the implemented detector on the example image. __Try out different parameters__ for the detector and __discuss the result__. Emphasize on the following aspects:
# - Basic idea and steps of the Canny edge detector.
# - Meaning of all parameters and their influence on the result.

# In[7]:


I = lab.imread3D('images/lena_wide.jpg')

# PARAMETERS:
sigma  = 2.0      # Std. dev. of gaussian for gradient computation [> 1.0]
m_strt = 10.0     # Gradient magnitude threshold to start a line [>= mag_cont]
m_cont = 0.5      # Gradient magnitude threshold to continue a line [>= 0.0]

E = canny_edge_detector(I, sigma, m_strt, m_cont)   # Apply edge detection

lab.imshow3D(I, E)                                  # Display result
print('left to right: original image | detected edge pixels')


# ### Discussion:
# 
# #### Basic idea and steps of the Canny edge detector.
# The Canny edge detector is an edge detection operator that uses a multi-stage algorithm to detect a wide range of edges in images. 
# Canny edge detector is depend on image gradients.
# Image gradients provides information about the magnitude and the directions.
# The Process of Canny edge detection algorithm can be broken down to 5 different steps:
# 1. Apply Gaussian filter to smooth the image in order to remove the noise and applying del_gaussian to detect the magnitude and direction, find the intensity gradients of the image
# 
# 3. Apply non-maximum suppression to get rid of spurious response to edge detection and non maximum supression magnitude is applied in the gradient direction.
# Non maximum pixels are transferring to 0 and with Local maximum pixel vice versa.
# 4. Apply double threshold to determine potential edges
# 5. Track edge by hysteresis: Finalize the detection of edges by suppressing all the other edges that are weak and not connected to strong edges. 
# It will check upper and lower threshold if edge pixels or not.
# 
# #### Meaning of all parameters and their influence on the result.
# Sigma is standard deviation of Gaussian which has purpose(gaussian) to smooth the image hence it will be less sensitive to noise.
# Thresholds: the use of two thresholds with hysteresis allows more flexibility than in a single-threshold approach, but general problems of thresholding approaches still apply. 
# If we increase the value of sigma, noise with high frequency is filtered then there will become blurr of edges as both edge and noise will be identified as high frequency signals. 
# A threshold set too high can miss important information.
# On the other hand, a threshold set too low will falsely identify irrelevant information (such as noise) as important.
# m_strt is gradient magnitude threshold for the beggining of line.
# When local maxima > Threshold, then set the pixel to our start edge candidates.
# It is the gradient magnitude which help to continue a line. 
# Threshold >>> large, might losing few edges in the case of Thresold <<<<< less then a lots of noise detected as comparision large threshold.

# ## Exercise 3: Line detection
# ### Exercise 3.1: Hough transformation
# __Complete__ the function below, that takes $M$, the 2D array of the gradient magnitudes and returns the 2D hough space representation where the x-axis represents the direction of the normal vector and the y-axis the distance of the line.

# In[8]:


# @jit(nopython=True, cache=True) # Uncomment to improve computation speed
def hough_space(M, m_min):
    h, w = M.shape
    
    max_dist = int((h**2 + w**2)**0.5 + 1)  # Maximum possible distance of a line to origin
    double_dist = max_dist * 2
    hough_space = np.zeros((180, double_dist), dtype=np.float64)
    
    for x in range(h):
        for y in range(w):
            magnitude = M[x,y]
            if magnitude < m_min:  # Additional threshold to improve computation speed
                continue
            
            for theta in range(180):
                
                # YOUR CODE GOES HERE 
                #
                # Step 1: compute the (minimum) distance 'd' between the origin and the line 
                d=x*math.cos(math.radians(theta))+y*math.sin(math.radians(theta))
                # Hints:  theta is given in degree here (functions usually require rad)
                
                hough_space[theta, int(np.round(d)+ max_dist )] += magnitude       
        
    return hough_space


# <font color='green' size='5'><b>OK</b></font>
# 
# <font color='orange'><b></b></font>

# ### Exercise 3.2: Evaluation
# The following cell will apply the hough transformation. The provided function `draw_best_lines()` will detect the local maximums in hough space and draw the corresponding lines to the image. It also returns the hough space as an image, where those local maximums are marked. __Discuss the results__ and emphasize on the following aspects and questions:
# - Basic idea of the hough transformation.
# - How could the hough transformation be used to detect ellipses? How many and which dimensions would the hough space have? 

# In[9]:


I = lab.imread3D('images/chip.jpg')
assert I.shape[2] == 1, "Hough line detection only valid for single channel images!"

# PARAMETERS:
sigma     = 2.5   # Scale for gradient computation   [> 1.0]
m_min     = 8.0   # Minimum magnitude                [>= 0.0]
num_lines = 9     # Number of best lines to show     [>= 0]

M, _ = lab.gaussian_gradients(I, sigma)       # Compute gradient magnitudes 
H    = hough_space(M[:, :, 0], m_min)         # Apply the hough transformation
I_lines, H_maxs = lab.draw_best_lines(I, H, num_lines, m_min)

M[M < m_min] = 0.0                            # Apply magnitude threshold (for visu.)
M_n = lab.normalize(M)

lab.imshow3D(I, M_n, I_lines)                 # Display results
print('left to right: original | thresholded gradient magnitudes | detected lines')
lab.imshow3D(H_maxs)                          # Display hough space with best lines
lab.imsave3D('chip_lins.jpg',I_lines)
lab.imsave3D('hough_space.jpg',H_maxs)
print('hough space with local maximums')


# #### Discussion
# 
# - Basic idea of the hough transformation:
# 
# The Hough transform takes a 2D array of the gradient magnitudes as input and attempts to locate edges placed as straight lines. <font color='green'><b>(The input to a Hough transform is normally an image that has been edge detected)</b></font>
# The idea of the Hough transform is, that every edge point in the edge map is transformed to all possible lines that could pass through that point. 
# Use the direction of the normal vector<font color='red'><b>[for what?] for the Hough transformation</b></font>. 
# In Hough transformation is use the direction of normal vector and the distance to the closest point on the straight line to represent it as point in parameters space and distance between the origin of coordinates. x(value of the point which is direction of the normal vector and y(value of point which is distance))
# The origin of the coordinates system and nearest point on the straight line distance show in the parameters space points.
# - How could the hough transformation be used to detect ellipses? 
#  If we have to deletct ellipses then we have to check every pixel with different lines and directions which are going through it Which represents in parameters space.
#  Each pixels represents by the curve which will be obtained and applying to every pixels. 
#  It can be applied to pixels with same line also hence each and every pixels.
#  Intersection of the curves in the parameters space represents line. (code line number 20)
# Maximum number of curve line used for the threshold detection for the determination of line is also known as relative maxima. 
#  
# - How many and which dimensions would the hough space have? 
# 
# 5 dimensions is required in the  detection of ellipse of an image does Hough space have. They are as 
# semimajor axis (a)
# semi-minor axis (b)
# origin/center of the coordinated y0 x0 
# rotation angle 
# They are having 5 dimension wrt 5 parameters as mention above.
# 

# ## Exercise 4: Segmentation
# ### Exercise 4.1: Region growing (basic)
# __Implement__ the function below, that performs a __basic__ region growing for segmentation of a single channel gray-value image $I$ (see the introduction slides). The second parameter $d_{max}$ determines the threshold for adding a pixel to a region w.r.t. the maximum absolute distance between the gray-value of the region and the potential pixel. The function should return the 2D-array $C$ where $C(x,y)$ is the region ID of pixel $(x,y)$. The first region should have the ID $0$.

# In[10]:


# @jit(nopython=True, cache=True) # Uncomment to improve computation speed
def region_growing_gray(I, d_max):
    h, w, d = I.shape
    assert d == 1, "Only single channel images supported!"
    
    C = np.ones((h, w), dtype=np.int64) * -1 # Initialize the region-id-map with -1
    
    # YOUR CODE GOES HERE
    #
    # Hints: take a look at the implementation of the 
    #        region growing algorithm for color images in lab.py. 
    #        Region smoothing, minimum region size and 
    #        updates of the region mean should NOT be implemented here!
    
    max_dist_sq = d_max**2 
    seeds = [(0, 0)][1:]
    for x in range(h):
        for y in range(w):
            seeds.append((x, y))
    current_seg_id = -1
    while len(seeds) > 0:
        sx, sy = seeds.pop()
        if C[sx, sy] >= 0:
            continue
        current_seg_id += 1
        C[sx, sy] = current_seg_id
        members = [(sx, sy)]
        sum_colors = np.copy(I[sx, sy])
        num_pixels = 1
        nearest_segment_distance = -1
        nearest_seg_id = -1
        mean_colors = np.copy(sum_colors)  # / num_pixels

        ### GROWING

        added_members = True
        while added_members:
            added_members = False
            for mx, my in members:
                neighbours = lab.get_valid_neighbours(h, w, mx, my)
                for nx, ny in neighbours:
                    ncol = I[nx, ny]
                    ns = C[nx, ny]
                    g_dist = np.sum(np.square(ncol - mean_colors))
                    if ns != current_seg_id and ns >= 0 and (
                            g_dist < nearest_segment_distance or nearest_seg_id == -1):
                        nearest_segment_distance = g_dist
                        nearest_seg_id = ns
                    if ns < 0 and g_dist < max_dist_sq:
                        C[nx, ny] = current_seg_id
                        num_pixels += 1
                        sum_colors += ncol
                        #mean_colors = sum_colors / num_pixels
                        members.append((nx, ny))
                        added_members = True
    
                  
                        
    return C


# <font color='red'><b>[Update of region mean should not be implemented in this exercise!]----------- I think it is done</b></font>

# ### Exercise 4.2: Evaluation
# The following cell will apply the region growing. The provided function `segments_to_image()` will create an image, where each class will have a random color. __Try different values__ for the gray-value threshold $d_{max}$ and __discuss the result__. Emphasize on the following aspects and questions:
# - Influence of the gray-value threshold $d_{max}$ on the result.
# - Observed problems of the basic region growing algorithm and ideas how to counteract them.
# - Which distance metrics could be used to perform region growing on a color image?  

# In[11]:


I = lab.imread3D('images/aerial_wide.jpg')  
assert I.shape[2] == 1, "Region growing only valid for single channel images!"

# PARAMETER:
d_max = 15 # Max. dist. between gray-value of neighbour and region mean [0-255]

C  = region_growing_gray(I, d_max)
Ic = lab.segments_to_image(C)

lab.imshow3D(I, Ic)         # Display segmentation result
print('left to right: original image | regions (random colorized)')


# ### Discussion
# 
# #### Influence of the gray-value threshold $d_{max}$ on the result.
# - too low value can result in an over-segmented image and reduce the np of classified regions as it is affected by its criteria.
# - too high value to an under-segmented image and we will obtain very less classes of regions of the images we have obtained.
# 
# #### Observed problems of the basic region growing algorithm and ideas how to counteract them.
# - It is very sensitive to noise as we can observe little regions inside some other big regions when they supposed to be considered big regions. <font color='green'><b>do you agree?</b></font>
# - Non smoothing varying region picture shows very high number of different region and oversegmentations hence it is not accurate and efficient to respective case.
# - ambiguities around edges of adjacent regions may not be resolved correctly
# - Different choices of seeds may give different segmentation results
# - Problems can occur if the (arbitrarily chosen) seed point lies on an edge
# - By setting up the mask <font color='red'><b> i do not understand it </b></font> to filtered(gaussian filter) the outlier pixels to remove the noise effect.
# Noise problem can be deal with is to set mask to filter out the outliers pixels for example Gaussian filter. So in the notes it has mention that noise problem can be avoided.
# - As per the lecture, we should have prior information of the image as the region is growing algorithem which is useful to set up the threshold and histogram help to the segmentation of the regions. 
# 
# #### Which distance metrics could be used to perform region growing on a color image?  
# Any one of the Distance matrix to be used in the color image if we have a reasonable number of colours enough for classifying the image into the regions. 
# So we can select the different region as per color wise.
# Image can be represents by RGB band with 3 grey scale images and apply algorithm to each and evry3 grey scale. Hence color as required paramerers distance matrix is used. 

# ### Exercise 4.3 Comparison with Watershed Transformation
# The following cell will apply a watershed transformation to perform the segmentation of the image. __Write a brief discussion__ which contains on the following aspects:
# - Main idea and steps of the watershed transformation.
# - Influence of the scale on the result.
# - Comparison of the result to the result of the region growing algorithm.

# In[12]:


I = lab.imread3D('images/aerial_wide.jpg')  
assert I.shape[2] == 1, "Region growing only valid for single channel images!"

# PARAMETER:
scale              = 1      # Standard deviation of the derivative of gaussian
new_seed_threshold = 5      # Threshold of gradient magnitude to start new seed

C  = lab.watershed_transform(I, scale, new_seed_threshold)
Ic = lab.segments_to_image(C)

lab.imshow3D(I, Ic)         # Display segmentation result
print('left to right: original image | regions (random colorized)')


# ### Discussion
# 
# #### Main idea and steps of the watershed transformation.
# Gradient magnitude image is interpreted as topographic surface
# Each pixels brightness is representd by its heights
# The topography is flooded starting at the global minimum
# Local minima are considered local wells -> initialise regions
# Prevent water from different catchment areas to mix -> watersheds
# The line related to the top of the watershed is considered as watershed line.
# Surfaces flood is the catchment basins if one can observe it carefully.
# Partition of image into:
# - Catchment areas -> Regions
# - Watershed lines -> region contours
# Regions are separated by borders with high gradient
# 
# As per research, this computation is known as Meyer Flooding computation.[google search too not sure it is correct ]
# Select and label the every pixels where the beginning of the flood.
# Obtain the gradient of the image and coodinated of the seed region of the flood.
# As per the priority wise the sequence is obtain of labelled pixels.
# Priority is decided by the gradient magnitude of the pixels.
# Lowest priority value to higher priority valuse is maintained in incrasing order hence all pixels are labelled and non marked put and keep exreacted.
# Remaining pixels are our results watershed lines.
# #### Influence of the scale on the result.
# When we are increasing the scale from 0.4 to 1 then increase of number of regions classified in the image which is represents by the standard deviation of the del_gaussian.
# Standard deviation is directly proportional to the more the regions we obtained.
# #### Comparison of the result to the result of the region growing algorithm.
# 
# Watershed algorithem with region growing algorithm, watershed is more accurate in the representing the borders of the classified regions as per different paramerters wise.
# Immediate variation is seen in the intensitied of the  pixels in the refion growing algo.
# The noise can we be eliminated but in very line-segment as we can vary the distance majorly in our case it can be increases in the distances. 
# 

# <font color='green' size='5'><b>OK</b></font>
# 
# <font color='orange'><b></b></font>

# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




