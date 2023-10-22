# Canny Edge Detection (fast vectorized implementation) + Fast multi-Seam Carving

This project implements Non-max Suppression, Hysteresis for Canny Edge detection and Seam Carving algorithms for image processing. The Canny Edge detection algorithm is implemented in a fast vectorized manner using torch (CPU only). The Seam Carving algorithm is implemented with

### Canny Edge-Detection

Non-max Suppression:
This algorithm is used to suppress non-maximum values in an image. It is commonly used in edge detection algorithms to thin out edges and reduce the number of pixels that represent an edge.

Hysteresis:
This algorithm is used to detect edges in an image. It works by applying two thresholds to an image and then connecting pixels that are above the high threshold and adjacent to pixels that are above the low threshold.

### Seam Carving:

This algorithm is used to resize images without distorting the important parts of the image. It works by finding the seams in an image that contain the least amount of energy and removing them. This process is repeated until the desired size is reached.

# Examples

### Canny Edge Detection:

|            Original             |          Non-Max Suppressed           |            Hysteresis Thresholding            |
| :-----------------------------: | :-----------------------------------: | :-------------------------------------------: |
|    ![OG](/Media/castle.jpg)     |    ![NMS](/Output/castle_NMS.jpg)     |    ![Hysteresis](/Output/castle_Hyst.jpg)     |
|     ![OG](/Media/core.jpg)      |     ![NMS](/Output/core_NMS.jpg)      |     ![Hysteresis](/Output/core_Hyst.jpg)      |
| ![OG](/Media/bowl-of-fruit.jpg) | ![NMS](/Output/bowl-of-fruit_NMS.jpg) | ![Hysteresis](/Output/bowl-of-fruit_Hyst.jpg) |

### Seam Carving:

|        Original        |        Seam Carved - Width         |       Seam Carved - Height        |
| :--------------------: | :--------------------------------: | :-------------------------------: |
| ![OG](/Media/York.jpg) | ![SC-X](/Output/York1200X1151.jpg) | ![SC-Y](/Output/York1728X720.jpg) |

# Usage

1. Clone the repository
2. Install the requirements
3. Run the CannyDetection_RunMe.py or SeamCarving_RunMe.py file. Image inputs and their configurations can be changed in the RunMe files.

# Implementation

### Canny Edge-Detection

1. Gradient calculation can be done using Sobel or Derivative of Gaussian (DoG) filters. Here, DoG is used by default.
2. NMS is vectorized quite efficiently using torch, and is done in a single loop.
3. Hysteresis is also vectorized using torch, and is done in a single loop.
4. Takes about 2-3 seconds to process a 1000x1000 image.
5. NMS Can benefit from GPU acceleration (Not planned right now).
6. Hysteresis cannot benefit from GPU acceleration unless concurrent BFS is implemented (Not planned right now).

### Seam-Carving

1. Energy calculation is done using Sobel filters.
2. Seam calculation is done using dynamic programming.
3. Seam removal is vectorized using torch, and is done in a single loop.
4. This is where significant speedup can be achieved by sacrificing quality. We can chose to remove multiple seams at once from a single Energy Accumulation Matrix, hence reducing the times the Energy Accumulation Matrix needs to be recalculated. This is done by setting the number of seams to be removed per calculation. However, this results in a loss of quality as the seams are not recalculated after each removal.
