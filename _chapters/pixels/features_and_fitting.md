---
title: Features and fitting
keywords: RANSAC, Harris, Local Invariant Features
order: 5
---

[//]: # (TODO!!!  ** overall intro **)


Table of contents

- [RANSAC](#ransac)
  - [Lines](#lines)
  - [Model Fitting](#model-fitting)
  - [Voting Based Fitting](#voting-based-fitting)
  - [Random Sample Consensus](#random-sample-consensus)
- [Local Invariant Features](#local-invariant-features)
  - [Motivaion](#lif-motivation)
  - [General Approach](#lif-general-approach)
  - [Requirements for Keypoint Localization](#lif-req-keypoint-loc)
- [Harris Corner Detector](#harris-corner-detector)

[//]: # (This is how you can make a comment that won't appear in the web page! It might be visible on some machines/browsers so use this only for development.)

[//]: # (Notice in the table of contents that [First Big Topic] matches #first-big-topic, except for all lowercase and spaces are replaced with dashes. This is important so that the table of contents links properly to the sections)

[//]: # (Leave this line here, but you can replace the name field with anything! It's used in the HTML structure of the page but isn't visible to users)


<a name='ransac'></a>
# RANSAC

In this lecture, we will introduce model fitting for line detection using the RANSAC (Random Sample Consensus) algorithm.

<a name='lines'></a>
## Lines

Straight lines characterize many objects around us. Here are few examples where line detection is useful:

<div class="fig figcenter">
  <img src="{{ site.baseurl }}/assets/pixels/lines-intro.png">
</div>

Looking at these examples and intuitively thinking of line detection, few questions arise.
- Once we have points that belong to a line, what is the line?
- How many lines exist?
- Which points belong to which line?

_Note:_ Edge detection is a technique that is used to identify boundaries. It does not provide information on the orientation of pixels. For example, whether these pixels form a line or corner.

<a name='model-fitting'></a>
## Model Fitting

We would like to form a higher-level succinct representation of the features in the image by grouping multiple features based on a simple model. This section focuses on lines described as one such model (i.e. line fitting) with edge points as features.

**Challenges**

As with any fitting problem, we need to take into account missing information, noise, and outliers. Our solution should be able to detect the best fitting line from edge points that could have clutter and noise. It should also be able to detect lines that bridge missing evidence.

<div class="fig figcenter">
  <img src="{{ site.baseurl }}/assets/pixels/line-fitting-challenges.png">
</div>

Least squares regression is a common technique to find a line of best fit for a dataset. However, least squares is prone to the negative effect of outliers. When dealing with images it is often the case that the number of outliers is large. A better approach is to find a set of inliers to initiate model fitting. In the next section, we'll explore such a technique that helps us find consistent matches.

<div class="fig figcenter">
  <img src="{{ site.baseurl }}/assets/pixels/line-fitting-outliers.png">
</div>

<a name='voting-based-fitting'></a>
## Voting-based Fitting

We could try to fit the model by exhaustively checking all combination of features. However, this approach is inefficient and bears a O(N<sup>2</sup>) time complexity.

Voting is a decent alternative. The idea is to have features vote for compatible models and to capture model parameters that form a majority. It turns out that this set of winning model parameters are not affected by votes from clutter and noisy features as their voting will be inconsistent with the consensus. The problem of missing observations can also be overlooked as the fitted model can span multiple fragments. In general, voting is considered to be robust to outliers and missing data.

[Hough Transform](../edge_detection) uses this voting strategy for detecting lines. Check out this video for a quick review of Hough transform:

<iframe style="display: block; margin: auto;" width="560" height="315" src="https://www.youtube.com/embed/4zHbI-fFIlI" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>

<!-- <<image 4.6 slide 11>> -->

<a name='random-sample-consensus'></a>
## Random Sample Consensus (RANSAC)

Both Hough transform and RANSAC rely on voting to arrive at the optimum model. The part where they differ is in how the model is chosen. As the name suggests, RANSAC introduces randomness in the model selection process. A number of models are proposed until one is found that is supported by a consensus of features (voters). Let's try to understand the algorithm.

**Algorithm**

RANSAC algorithm was developed by Fischler and Bolles in 1981. Here's the gist:

1. Randomly sample a group of points required to fit the model
2. Find model parameters from this sample
3. Calculate the votes from the fraction of inliers within a predetermined threshold of the model
4. Repeat steps 1-3 until model is found with high confidence

**Algorithm Pseudocode**

<div class="fig figcenter">
  <img src="{{ site.baseurl }}/assets/pixels/ransac-pseudocode.png">
</div>

**Model Parameters**

The algorithm can be tuned with the following parameters:

- $n$ – minimum number of points required to estimate model parameters
- $k$ – maximum number of iterations
- $t$ – distance threshold to determine points that fit well to model
- $d$ – number of close points required to assert a model fits well

The parameter that needs most attention is _k_ – number of iterations. Next, we'll see how to determine its value.

**Estimate number of iterations**

Random sampling requires a minimum number of samples to be drawn to provide a high confidence estimate of parameters. The number of samples largely depends on outliers and sample size. In other words, the number of iterations is directly proportional to noise in the data and complexity of the model. This can be described by the following probabilities:

$$ \begin{equation} P_f = (1 - W^n)^k = 1 - p \end{equation} $$

where $k$ is the number of samples,  $n$ is the minimum number of points required to estimate model parameters (e.g. line has 2 parameters), $W$ is the fraction of inliers, and $P_f$ represents probability of _k_ samples failing.

Solving for $k$ gives us the minimum number of samples required to keep the failure rate low.

$$ \begin{equation} k = \frac{\log(1 - p)}{\log(1 - W^n)} \end{equation} $$

The table below shows the number of samples required for different choices of noise and model size. It's evident that $k$ increases sharply as $n$ increases.

<div class="fig figcenter">
  <img src="{{ site.baseurl }}/assets/pixels/RANSAC-number-of-iterations.png">
</div>

**Refined RANSAC**

**Analysis**

**Summary**

<a name='local-invariant-features'></a>
# Local Invariant Features

**Introduction**

As we enter our segment on local invariant features, it is important to understand where we are in terms of image detection and computer vision. As we will focus on specific features and  pixels within images, we are diving into an area that can cause problems. For example, if we have one picture of the pentagon as a template and this picture is in black and white, will we be able to use computer vision to find the pentagon when we see it from a distance on a plane? What about when we look at two pictures of the same thing but at different angles? How can we make it so the algorithm understands we are looking at images of the same thing? We can solve this problem using local invariant features within our image matching. 

<a name='lif-motivation'></a>
**Motivation**

Global representations of images have major limitations in identifying small objects and image matching, due to potential differences in perspective, scale, and more between images. Instead, we can describe and match local regions. By using local features, we are able to identify objects despite occlusion, articulation, and intra-category variations.

This way, we can solve the issue of image matching and procude an algorithm which is much more effective at detecting which features of each image align regardless of differences in angle, color, brightness, etc.

<a name='lif-general-approach'></a>
**General Approach**

Suppose we have two input images. The task is to identify whther the two images are of the same object. Using local invariant features, the general approach is summarized in the following steps.

1. Find a set of distinctive key points in two given images. For example, these key points can be edges or corners.

2. Define a region around each keypoint. Typically, a square region is selected around each keypoint.

3. Extract and normalize the region content. This step accounts for shift-variance, scale-variance, as well as changes in lighting conditions.

4. Compute a local descriptor from the normalized region. The descriptor takes the form of a vector or a function that describes each region.

5. Match the keypoints in the input images. This is done by finding the similarities between each local descriptor in both images and matching the most similar keypoints.

The figure below summarizes each step and provides a visual description of the general approach.

<div class="fig figcenter">
  <img src="{{ site.baseurl }}/assets/pixels/lif-approach.jpg">
</div>

The process of matching keypoints between two images is called **keypoint localization**.

<a name='lif-req-keypoint-loc'></a>
**Requirements for Keypoint Localization**

There are many approaches for keypoint localization. However, a good approach is able to achieve the following goals:

1. Detect the same point independently in both images: must be able to find the same points independently in both images (need a **repeatable** keypoint detector to run on both images independently).

2. For each point, correctly recognize the corresponding point in the opposite image (need a **reliable** and **distinctive** descriptor to correctly detect these points).

3. Needs to be invariant to **geometric** (scaling, affine, or out-of-plane transformations, rotation) and **photometric** (lighting, noise, blur, etc.) changes between images

With these requirements, the region extraction needs to be repeatable and accurate. This way, it will be invariant to translation, rotation, and scale changes. In addition, it will be robust and covariant to affine transformations, lighting variations, noise, blur, and quantization.

<a name='harris-corner-detector'></a>
# Harris Corner Detector

Recall that we have the following goals for keypoint localization:
- Repeatable detection: keypoint detector can run on different images independently and detect the same keypoints
- Precise localization: accurately detect keypoints in images at the correct location
- Interesting content: areas with strong change (not in homogeneous regions, which are not distinctive and hard to match)

**Corners as Distinctive Interest Points**
- Key property of corners: In the region around a corner, the image gradient has two or more dominant directions (for instance, perpendicular gradients would represent a 90 degree angled corner). 
- Corners are repeatable and distinctive (can be easily seen from multiple view points, and are very different from their neighbors)

**Design Criteria**

1. Locality: when looking through a small window, we should be able to easily recognize the corner point.

2. Good localization: shifting the window in any direction should give a large change in intensity.

**Examples**
<div class="fig figcenter">
  <img src="{{ site.baseurl }}/assets/pixels/flat-edge-corner.png">
</div>

"Flat" region: no change in any direction.
- Small $\sum I_x^2$ and small $\sum I_y^2$. 

"Edge": no change along the edge direction, but change along the perpendicular dimension.
- Small $\sum I_x^2$ and large $\sum I_y^2$ for a horizonal edge.
- Large $\sum I_x^2$ and small $\sum I_y^2$ for a vertical edge.

"Corner": significant change in all directions.
- Large $\sum I_x^2$ and large $\sum I_y^2$.
- However, multiple corners that are connected by an index produce issues because we cannot conclude the size of $\sum I_x^2$ and $\sum I_y^2$.


**Harris Detector Formulation**

1. Localize patches that result in large change of intensity when shifted in any direction.

2. When we shift by $[u, v]$, the intesity change at the center pixel is measured as intensity difference: 

$$ \begin{equation} I(x + u, y + v) - I(x, y) \end{equation} $$

- This intensity difference measurement is for one single point, but we need to accumulate over the patch around that point as well. Therefore, when we shift by $[u, v]$, the change in intensity for the patch is: 
<div class="fig figcenter">
  <img src="{{ site.baseurl }}/assets/pixels/harris-formulation.png">
</div>

  - $\sum_{x,y}$ is the sum over the entire window / patch
  - $w(x, y)$ is the window function
  - $I(x + u, y + v)$ is the shifted intensity of a given pixel
  - $I(x, y)$ is the original intensity of a given pixel
  - $I(x + u, y + v) - I(x, y)$ is the total intensity change for a given pixel

- Approximating the change in patch intensity (with Taylor expansion):
$$
E(u, v) \approx
\begin{bmatrix}
u & v
\end{bmatrix}
M
\begin{bmatrix}
u \\ v
\end{bmatrix}
$$
  - where $M$ is a 2x2 matrix computed from image derivatives:
  <div class="fig figcenter">
  <img src="{{ site.baseurl }}/assets/pixels/matrixM.png">
  </div>
  - graphical intuition for image gradients $I_x, I_y, I_x I_y$:
  <div class="fig figcenter">
  <img src="{{ site.baseurl }}/assets/pixels/image-derivative.png">
  </div>
  - derivation: 
  $$
  I(x+u, y+v) \approx I(x, y) + u I_x + v I_y \; \text{(Taylor expansion)}
  $$
  $$
  \begin{aligned}
  E(u, v) &\approx \sum_{x, y} w(x, y) \left[ u I_x + v I_y \right]^2 \\
  &= \sum_{x, y} w(x, y) \left| 
  \begin{bmatrix} I_x & I_y \end{bmatrix} 
  \begin{bmatrix} u \\ v \end{bmatrix} \right|^2 \\
  &= \sum_{x, y} w(x, y) \left( 
  \begin{bmatrix} I_x & I_y \end{bmatrix} 
  \begin{bmatrix} u \\ v \end{bmatrix} \right)^T
  \left( \begin{bmatrix} I_x & I_y \end{bmatrix} 
  \begin{bmatrix} u \\ v \end{bmatrix} \right) \\
  &= \sum_{x, y} w(x, y) 
  \begin{bmatrix} u & v \end{bmatrix} 
  \begin{bmatrix} I_x \\ I_y \end{bmatrix}
  \begin{bmatrix} I_x & I_y \end{bmatrix} 
  \begin{bmatrix} u \\ v \end{bmatrix}  \\
  &= \sum_{x, y} w(x, y) 
  \begin{bmatrix} u & v \end{bmatrix} 
  \begin{bmatrix}
  I_x^2 & I_x I_y \\
  I_x I_y & I_y^2
  \end{bmatrix}
  \begin{bmatrix} u \\ v \end{bmatrix}  \\
  &= \begin{bmatrix} u & v \end{bmatrix} \left\{
  \sum_{x, y} w(x, y) 
  \begin{bmatrix}
  I_x^2 & I_x I_y \\
  I_x I_y & I_y^2
  \end{bmatrix} \right\}
  \begin{bmatrix} u \\ v \end{bmatrix}  \\
  &= \begin{bmatrix} u & v \end{bmatrix} M
  \begin{bmatrix} u \\ v \end{bmatrix}
  \end{aligned}
  $$

- Meaning behind matrix $M$:
  <div class="fig figcenter">
  <img src="{{ site.baseurl }}/assets/pixels/axis-aligned-M.png">
  </div>

  - Consider an axis aligned corner, and assume $w(x, y) = 1$
  $$
  M = \sum_{x, y} 
  \begin{bmatrix}
  I_x^2 & I_x I_y \\
  I_x I_y & I_y^2
  \end{bmatrix}
  = \begin{bmatrix}
  \sum I_x^2 & \sum I_x I_y \\
  \sum I_x I_y & \sum I_y^2
  \end{bmatrix}
  = \begin{bmatrix}
  \lambda_1 & 0 \\
  0 & \lambda_2
  \end{bmatrix}
  $$
  Pixels on the vertical edge will have $I_y = 0$ and $I_x^2 >> 0$ (marked in green), therefore only contributing to the $\sum I_x^2$ element in matrix $M$. Similarly, pixels on the horizontal edge will have $I_x = 0$ and $I_y^2 >> 0$ (marked in orange), therefore only contributing to the $\sum I_y^2$ element in matrix $M$. Other pixels have $I_x = 0$ and $I_y = 0$ and do not contribute to the sums in the matrix. The only non-zero elements in matrix $M$ are the diagonal elements $\sum I_x^2 = \lambda_1$ and $\sum I_y^2 = \lambda_2$.
  - Our window contains an axis aligned corner if and only if both $\lambda_1$ and $\lambda_2$ are large. If either $\lambda$ is close to 0, then the window does not contain an axis aligned corner.
  - In the general case, we can decompose the symmetric matrix $M$ as 
  $$
  M = \begin{bmatrix}
  \sum I_x^2 & \sum I_x I_y \\
  \sum I_x I_y & \sum I_y^2
  \end{bmatrix}
  = R^{-1} \begin{bmatrix}
  \lambda_1 & 0 \\
  0 & \lambda_2
  \end{bmatrix} R \; \text{(eigenvalue decomposition)}
  $$
  <div class="fig figcenter">
  <img src="{{ site.baseurl }}/assets/pixels/rotated-M.png">
  </div>
  We can interpret $M$ as an ellipse with its axis length determined by the eigenvalues $\lambda_1$ and $\lambda_2$; and its orientation determined by $R$.
  A rotated corner will have the same eigenvalues as its non-rotated version, and the rotation will be captured by the rotation matrix $R$.
  - Interpreting the eigenvalues: 
  <div class="fig figcenter">
    <img src="{{ site.baseurl }}/assets/pixels/eigenvalue_harris.png">
  </div>

  Comparing eigenvalues gives us a good indicator of distinct features of an image: 
  - $\lambda_2 >> \lambda_1$ or $\lambda_1 >> \lambda_2$: **Edge**
  - $\lambda_2, \lambda_1 \approx 0$: **Flat**
  - $\lambda_2 \approx \lambda_1$: **Corner**

- Because calculating eigenvalues, especially for large and hi-res images, is
   computationally expensive, the **Corner Response Function** is a widely 
   used, fast alternative: 

   $$ 
   \theta = det(M) - \alpha trace(M)^2 = \lambda_1 \lambda_2 - \alpha (\lambda_1 + \lambda_2)^2
   $$

   where $\alpha$ is a constant  (~$0.04 - 0.06)$. 


- Window Function
<div class="fig figcenter">
  <img src="{{ site.baseurl }}/assets/pixels/window-function.png">
</div>

  1. Uniform window: 
  - sum over square window
  $$
  M = \sum_{x, y} \begin{bmatrix}
  I_x^2 & I_x I_y \\
  I_x I_y & I_y^2
  \end{bmatrix}
  $$
  - problem: not rotation invariant

  2. Smooth with Gaussian
  - Gaussian already performs weighted sum
  $$
  M = g( \sigma ) * \begin{bmatrix}
  I_x^2 & I_x I_y \\
  I_x I_y & I_y^2
  \end{bmatrix}
  $$
  - result is rotation invariant


**Harris Detector Implementation**
<div class="fig figcenter">
  <img src="{{ site.baseurl }}/assets/pixels/harris-summary.png">
</div>
<div class="fig figcenter">
  <img src="{{ site.baseurl }}/assets/pixels/harris-summary2.png">
</div>

1. Compute image derivatives $\Rightarrow I_x, I_y$
2. Compute the square of image derivatives $\Rightarrow I_x^2, I_y^2, I_x I_y$
3. Apply Gaussian filter $g(\sigma_I)$ $\Rightarrow g(I_x^2), g(I_y^2), g(I_x I_y)$
4. Compute corner response function:
  - compute matrix $M$ (aka. second moment matrix / autocorrelation matrix)
  $$
  M(\sigma_I, \sigma_D) = \begin{bmatrix}
  I_x^2(\sigma_D) & I_x I_y(\sigma_D) \\
  I_x I_y(\sigma_D) & I_y^2(\sigma_D)
  \end{bmatrix}
  $$
  - $\sigma_D$: for Gaussian in the derivative calculation
  - $\sigma_I$: for Gaussian in the windowing function
  - $$\theta = \text{det}[M(\sigma_I, \sigma_D)] - \alpha \text{trace}[M(\sigma_I, \sigma_D)]^2 = g(I_x^2)g(I_y^2) - [g(I_x I_y)]^2 - \alpha [g(I_x^2) + g(I_y^2)]^2$$
5. Perform non-maximum suppression

  <div class="fig figcenter">
    <img src="{{ site.baseurl }}/assets/pixels/harris_response.png">
  </div>
  An example of Harris detection of an image.

**Scale Invariance**
- The Harris corner detector is *translation invariant* and *rotation invariant* (when used with a Gaussian kernel), but it is *not* **scale-invariant**. 

  <div class="fig figcenter">
    <img src="{{ site.baseurl }}/assets/pixels/sift_scale_invariant.png">
  </div>

  As shown here, the Harris detector is correctly able to recognize a corner with a small set window, but can no longer identify gradients when the image is enlarged and instead incorrectly identifies edges. This is ultimately why other, scale-invariant feature detectors are used for these purposes. 
