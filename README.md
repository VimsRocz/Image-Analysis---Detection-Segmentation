# Image-Analysis-Detection-Segmentation
This repository contains Python code for image analysis tasks, focusing on point detection using the Harris, Foerstner operators, as well as edge pixel detection using the Canny edge detector,line detection using the Hough transformation, and image segmentation using region growing and watershed transformation.

This repository contains implementations and discussions of fundamental image processing techniques. The exercises cover point detection, edge detection, line detection, and segmentation.

### Exercise 1: Point Detection
- Implementations of the Harris and Foerstner operators for corner detection.
- Explore parameters such as differentiation scale, integration scale, cornerness parameter, and minimum isotropy.

### Exercise 2: Canny Edge Detector
- Implementation and discussion of the Canny edge detection algorithm.
- Explore parameters such as sigma, gradient magnitude thresholds, and their influence on the result.

### Exercise 3: Line Detection
- Implementation of the Hough transformation for line detection.
- Evaluation of results and discussion on the basic idea of the Hough transformation.
- Exploration of extending the Hough transformation for detecting ellipses.

### Exercise 4: Segmentation
- Implementation of region growing for image segmentation.
- Evaluation of different gray-value thresholds ($d_{max}$) and discussion on their influence on the segmentation result.
- Comparison between region growing and watershed transformation for image segmentation.
- Discussion on the main idea and steps of the watershed transformation and its influence on the segmentation result.

## Features
- **Point Detection**: Implementations of the Harris and Foerstner operators for corner detection, enabling scale-adapted detection of interest points in images.
- **Edge Pixel Detection**: Implementation of the Canny edge detector, which identifies edges in images by detecting local maxima in gradient magnitudes and applying hysteresis to determine edge pixels.

## Usage
- **Point Detection**: Utilize the `scale_adapted_harris_foerstner` function to detect corners in images. Adjust parameters such as differentiation scale, integration scale, cornerness parameter, and minimum isotropy as needed.
- **Edge Pixel Detection**: Utilize the `canny_edge_detector` function to detect edges in images using the Canny edge detector. Specify parameters such as sigma for Gaussian smoothing, minimum gradient magnitude to start an edge, and minimum gradient magnitude to continue an edge.

## Conclusion
Through these exercises, various fundamental image processing techniques and their parameters have been explored and evaluated. The discussions provide insights into the strengths, limitations, and practical considerations of each technique, helping to deepen understanding and proficiency in image processing algorithms.
