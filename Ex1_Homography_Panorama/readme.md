![](Aspose.Words.868d2d4b-a55a-436a-8712-0ab0f17d8ccf.001.png)

**Computer Vision**

**Homography & Panorama**

**Preparatory Steps**

In the images below we can see the matched points vs the perfect match points on the source and destination images.

![](Aspose.Words.868d2d4b-a55a-436a-8712-0ab0f17d8ccf.002.png)

**Part A1: Homography Computation**

1. Given a Cartesian coordinate point pw=xw,yw,zw from the 3D space of the world, it is possible to represent the point in Homogeneous coordinate pw=xw,yw,zw using the following transformation:

xwywzw=f0000f000010xwywzw1

This transformation can be also denoted by the following equations

xw=fxw;       yw=fyw;       zw=zw                  for non-zero scalar, f

This point can be projected onto image plain that received by a pinhole camera according to the following transformation that we developed in class 

p=x,y    →    p=xy1

This point can be projected onto a new image plain (destination image plain) by a projective transformation. Such that

Pdst=HPsrc

Where H is the Homography matrix that given by:

H=h11h12h13h21h22h23h31h32h33

Since we are in Homogeneous coordinate, we can constraint dividing H by h33 so we will get the following:

uvw=h11h12h13h21h22h23h31h321xy1

This matrix is taking into account the intrinsic and extrinsic parameters of the camera, real world constrains (i.e., where the camera is located in the world and its orientation) and the fact that image plain doesn’t have z-element, so we can transform the image (“**source image**”) to another image plain (“**destination image**”) by the Homography transformation (“**projective transformation**”).

This transformation gives us the pixel in the destination image, but in Homogenous coordinate. 

u=h11x+h12y+h13

v=h21x+h22y+h23

w=h31x+h32y+1

Hence, we should divide each coordinate in w, finally we will get the pixel in the destination image in Cartesian coordinate by:

u=uw=h11x+h12y+h13h31x+h32y+1 

v=vw=h21x+h22y+h23h31x+h32y+1

` `w=ww=1

Let us develop the linear equation matrix system for u

h31x+h32y+1u=h11x+h12y+h13

h11x+h12y+h13-uh31x-uh32y-u=0

h≜h11h12h13h21h22h23h31h321T

au≜xy1000-ux-uy-uT

In a similar way for v

av≜000xy1-vx-vy-vT

Since we have 8 un-known variables, we can define a matrix A for 4 different known points, so h will be the eigen vector if the homogenic linear equation

A∙h=0,     where A=au,1av,1⋮au,4av,4 

Now, we can see that by solving this homogeneous equations’ system, we will get the projective matrix H.

Since the system is not "noise-free" (and in case more pairs are used), H can be the solution of the optimization problem

argminh A∙h2 s.t. |h|=1

Note that

` `A∙h2=AhTAh=hTATAh 

And denote by {λi}i=19, {ei}i=19 the eigen values and eigen vectors, correspondingly, of ATA such that ATAei= λiei.

Since ATA  is a PSD matrix we know that

λ1 ≥…≥λ9≥0





And due to the fact that {ei}i=19 forms an orthonormal basis, h can be expressed as 

h= i=19αiei

Where the norm is,

h|= 1=|h|2=i=19αiei|=i=19αi2 

Therefore,

` `A∙h2= hTATAh= i=19αieiTATAi=19αiei

After rearrangement and using the fact that ATAei= λiei we get that

` `A∙h2= i=19αi2λi ≥ λ1i=19αi2= λ1=e1TATAe1=Ae12

yields hopt= e1


In summary, since we have noises in the system the system is not homogenic so we will find the optimal solution of

AT∙A

In order to get the solution, we used the mathematic method SVD (Singular Values Decomposition) to get the eigenvectors matrix while the last vector of this matrix should give us the eigenvector for the smallest eigenvalue.

1. See implementation of **compute\_homography\_naive** in **ex1\_student\_solution**



1. Running the function from section 2 on the perfect match points file (matches\_perfect.mat) retrieves the following normalized Homography matrix, Hnaive. 

Hnaive=1.43457214e+00  2.10443232e-01 -1.27718679e+031.34265155e-02  1.34706123e+00-1.60455874e+01 3.79279298e-045.56523148e-051.00000000e+00

Snapshot for documentation:

![](Aspose.Words.868d2d4b-a55a-436a-8712-0ab0f17d8ccf.003.png)

**Part A2: Forward Mapping - Slow and Fast**

1. See implementation of **compute\_forward\_homography\_slow** in **ex1\_student\_solution**

The slow commutation took ~9s.

![](Aspose.Words.868d2d4b-a55a-436a-8712-0ab0f17d8ccf.004.png)

The function retrieves the following image.

![](Aspose.Words.868d2d4b-a55a-436a-8712-0ab0f17d8ccf.005.png)

1. See implementation of **compute\_forward\_homography\_fast** in **ex1\_student\_solution** 


The computation time of this method is roughly ~90ms.

![](Aspose.Words.868d2d4b-a55a-436a-8712-0ab0f17d8ccf.006.png)

The function retrieves the following image.

![](Aspose.Words.868d2d4b-a55a-436a-8712-0ab0f17d8ccf.007.png)

It can clearly be observed that the same image was produced by both of the methods of Forward Homography implementation, however the Fast method yields it within much less computation time.

1. Forward Mapping evaluates every pixel in the destination image with the value of its corresponding pixel in the source image, where the corresponding pair of pixels are calculated using the projection matrix H. Therefore, we may encounter the following two problems when using it:
   1. Integer coordinates might be mapped to non-integer coordinates, when the new calculated pixel in the destination image has to be rounded in order to get a valid pixel coordinate.
   1. When the source and destination images are not of the same size or orientation, some information of the source image might be distorted, or missing in the destination image, so that some black pixels might appear in the transformed image.

These issues are seen in the results we yield: dark pixels and artificial edges that are missing in the original image.

![](Aspose.Words.868d2d4b-a55a-436a-8712-0ab0f17d8ccf.008.png)

1. When we used imperfect match points (matches.mat), we got different Homography matrix. 

![](Aspose.Words.868d2d4b-a55a-436a-8712-0ab0f17d8ccf.009.png)

This projection matrix results in an unclear and messy image that looks even partially upside-down.

![](Aspose.Words.868d2d4b-a55a-436a-8712-0ab0f17d8ccf.010.png)

As we learned in the class, we know that noises affect the Homography model a lot, so we can conclude that the non-matched points, from imperfect match points file, are the noises that changed our Homography matrix and the result image. In addition, we can see that in the left side of the image the “holes”, that caused by the forward mapping process, were increased.


**

**Part B: Dealing with Outliers**

1. See implementation of **test\_homography** in **ex1\_student\_solution** 

The test Homography function calculates the following 2 parameters.

- Probability that the projection of a single pixel from the source image will be an inlier in the destination image, based on imperfect match points.
- The mean squared distance between projected pixels from the source image and their actual location in the destination image of inliers.

Here we can see our results, while the probability is 16% and the MSE error is the other value (456 in pixels units).

![](Aspose.Words.868d2d4b-a55a-436a-8712-0ab0f17d8ccf.011.png) 

1. See implementation of **meet\_the\_model\_points** in **ex1\_student\_solution**

1. For the following RANSAC parameters, let us calculate the given 2 cases.

w-inliers percent

t-maximal error

p-probability of the algorithm will success

d-the minimal probability of points meeting the model 

k-number of iteration

p=1-1-wnk        →          k=log1-plog1-wn

Given: N=30,  w=80%

As we showed in part A, at least 4 points are required in order to find the Homography coefficients. Hence,

p=1-1-0.84k


We will calculate k  in the following cases for by assuming independent random points.

Case I: p=90%

k=log1-0.9log1-0.84=4.37→ for this case we need mimum 5 iterations

Case II: p=99%

k=log1-0.99log1-0.84=8.74→ for this case we need mimum 9 iterations

For both of the cases introduced in the question, we need at least 9 iterations with 4 random points for each.

In order to cover all the options for selecting n=4 points out of the total 30 matched points, we need n=304=30!4!30-4!=27405.  

1. See implementation of **compute\_homography** in **ex1\_student\_solution**

1. The obtained RANSAC Homography matrix and its output image are given below.

![](Aspose.Words.868d2d4b-a55a-436a-8712-0ab0f17d8ccf.012.png)

Above results show how well the RANSAC algorithm deals with outliers: the projective matrix's coefficients are close to those we received for the perfectly matching points in the previous sections. The MSE distance is also significantly decreased.

All the mentioned above can explains the similarity of the RANSAC output image (below), compared to the perfect match transformation images.

![](Aspose.Words.868d2d4b-a55a-436a-8712-0ab0f17d8ccf.013.png)**

**Part C: Panorama Creation**

1. See implementation of **compute\_backward\_mapping** in **ex1\_student\_solution** 

1. See implementation of **add\_translation\_to\_backward\_homography** in **ex1\_student\_solution** 

1. See implementation of **panorama** in **ex1\_student\_solution**

1. Panorama output image

![](Aspose.Words.868d2d4b-a55a-436a-8712-0ab0f17d8ccf.014.png)

The images are "stitched" thanks to the overlapping sections where the inliers are.**

**


1. In order to get acceptable results, we chose the matching points that marked in red below.

![](Aspose.Words.868d2d4b-a55a-436a-8712-0ab0f17d8ccf.015.png)

In addition, we changed the maximal error and decimation factor as shown below.

![](Aspose.Words.868d2d4b-a55a-436a-8712-0ab0f17d8ccf.016.png)

![](Aspose.Words.868d2d4b-a55a-436a-8712-0ab0f17d8ccf.017.png)

And then ran the script and got the following results. While we can see the we got the same results with similar phenomena as we got previously above.

![](Aspose.Words.868d2d4b-a55a-436a-8712-0ab0f17d8ccf.018.png)

![](Aspose.Words.868d2d4b-a55a-436a-8712-0ab0f17d8ccf.019.png)


![](Aspose.Words.868d2d4b-a55a-436a-8712-0ab0f17d8ccf.020.png)

![](Aspose.Words.868d2d4b-a55a-436a-8712-0ab0f17d8ccf.021.png)



