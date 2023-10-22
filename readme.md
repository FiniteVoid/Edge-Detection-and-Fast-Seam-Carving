# Canny Edge Detection (fast vectorized implementation) + Fast multi-Seam Carving

This project implements Non-max Suppression, Hysteresis for Canny Edge detection and Seam Carving algorithms for image processing. The Canny Edge detection algorithm is implemented in a fast vectorized manner using torch (CPU only). The Seam Carving algorithm is implemented with

Non-max Suppression:
This algorithm is used to suppress non-maximum values in an image. It is commonly used in edge detection algorithms to thin out edges and reduce the number of pixels that represent an edge.

Hysteresis:
This algorithm is used to detect edges in an image. It works by applying two thresholds to an image and then connecting pixels that are above the high threshold and adjacent to pixels that are above the low threshold.

Seam Carving:
This algorithm is used to resize images without distorting the important parts of the image. It works by finding the seams in an image that contain the least amount of energy and removing them. This process is repeated until the desired size is reached.

# Examples

### Canny Detection:

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
