# **Finding Lane Lines on the Road** 

---

## Goals

 - Detect and draw a red overlay lines on the lane markers from images.
 - A single overlay marker must cover the entire stretch of markers of a lane without disconnects.
 - Given a video clip of lanes, the output video should have the red line overlay for every frame of the video.


---

### Reflections

### 1. Current Pipeline
This program works on the basic assumption that the polarity of the slope will always negative for left lane and positive for right lane. 

### Steps

 1. Convert Yellow lines to White
 Create a mask to extract yellow lines. An hsv upper and lower threshold is required to  create the mask. The yellow line is replaced with white line and superimposed on the sample image.
 2. Grayscale the image
 Convert the sample image to a grayscale. Further steps will not work on RGB image. 
 3. Apply Gaussian blur
 Some of the jitters are removed on the canny edges on a smoothened image. A kernel size of 5 is usually used. 
 4. Canny edges
 The process of finding edges requires you to set threshold limits between which the edges are retained on the output image.
 5. Region of Interest
 We need to retain only the lanes from the edges. A vertices array with quadrilateral coordinate points are required to be passed. The pixels outside the region of interest is masked out.
 6. Hough lines
 A list of lines are extracted from the canny edges image through hough transform. These parameters play a crucial role in picking up the lanes lines and avoiding irrelevant lines and noise.
	 1. theta: The resolution of the parameter theta in radians. We use 1 degree (CV_PI/180)
	 2. rho : The resolution of the parameter r in pixels. We use 1 pixel.
	 3. threshold: The minimum number of intersections to “detect” a line.
	 4. minLinLength: The minimum number of points that can form a line. Lines with less than this number of points are disregarded.
	 5. maxLineGap: The maximum gap between two points to be considered in the same line.
 7. Average slopes from hough lines
 From Cartesian coordinate information from hough lines, we calculate slopes for all lines. We consider the negative slope lines to be from the left lane marks and the positive from the right lane lines and list them separately as *left_lane* list and right_lane list. We calculate the average slope and intercept of the left_lane list and *right_lane* list. 
 8. Draw lanes
 Equipped with *left_lane* slope, intercept and *right_lane* slope and intercept, we calculate the end points of our red line overlay. A horizon is a coordinate below which we detect lane marks. A suitable horizon in our sample images was observed to be 320px.  We draw left and right lane lines over the sample image.

[//]: # (Image References)

[image1]: ./examples/laneLines_thirdPass.jpg "It should look like this"


### 2. Shortcomings

 1. Our premise that slopes will always be of opposite polarity may not hold good at the entire stretch of sharp curves in which case our algorithm will not work.
 2. Since we find the average of hough lines, a set of similar slopes on our sample images help our algorithm perform well. In the curves, there will be many hough lines with inconsistent slopes. Averaging these slopes will give a very bad estimate of our line.
 3. When there are not enough strips the resultant line has jitters.
 4. Our choice of horizon need to be dynamic based on the flatness/elevation of the road.

### 3. Possible improvements

 1. Instead of separating right lane and left lane with polarity of slopes, We may be able to use "similar slopes" and "proximity of coordinates of adjacent hough lines" combination to get the same result with additional stability in curved roads. 
 2. Instead of drawing a single mean line, it may be more stable to connect adjacent lines with tangent lines to form a collection of joined lines with varying slopes making it flexible to curves.



