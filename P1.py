
# coding: utf-8

# # **Finding Lane Lines on the Road** 
# ***
# In this project, you will use the tools you learned about in the lesson to identify lane lines on the road.  You can develop your pipeline on a series of individual images, and later apply the result to a video stream (really just a series of images). Check out the video clip "raw-lines-example.mp4" (also contained in this repository) to see what the output should look like after using the helper functions below. 
# 
# Once you have a result that looks roughly like "raw-lines-example.mp4", you'll need to get creative and try to average and/or extrapolate the line segments you've detected to map out the full extent of the lane lines.  You can see an example of the result you're going for in the video "P1_example.mp4".  Ultimately, you would like to draw just one line for the left side of the lane, and one for the right.
# 
# ---
# Let's have a look at our first image called 'test_images/solidWhiteRight.jpg'.  Run the 2 cells below (hit Shift-Enter or the "play" button above) to display the image.
# 
# **Note** If, at any point, you encounter frozen display windows or other confounding issues, you can always start again with a clean slate by going to the "Kernel" menu above and selecting "Restart & Clear Output".
# 
# ---

# **The tools you have are color selection, region of interest selection, grayscaling, Gaussian smoothing, Canny Edge Detection and Hough Tranform line detection.  You  are also free to explore and try other techniques that were not presented in the lesson.  Your goal is piece together a pipeline to detect the line segments in the image, then average/extrapolate them and draw them onto the image for display (as below).  Once you have a working pipeline, try it out on the video stream below.**
# 
# ---
# 
# <figure>
#  <img src="line-segments-example.jpg" width="380" alt="Combined Image" />
#  <figcaption>
#  <p></p> 
#  <p style="text-align: center;"> Your output should look something like this (above) after detecting line segments using the helper functions below </p> 
#  </figcaption>
# </figure>
#  <p></p> 
# <figure>
#  <img src="laneLines_thirdPass.jpg" width="380" alt="Combined Image" />
#  <figcaption>
#  <p></p> 
#  <p style="text-align: center;"> Your goal is to connect/average/extrapolate line segments to get output like this</p> 
#  </figcaption>
# </figure>

# In[3]:

#importing some useful packages
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2
get_ipython().magic('matplotlib inline')


# In[4]:

#reading in an image
image = mpimg.imread('test_images/solidWhiteRight.jpg')
#printing out some stats and plotting
print('This image is:', type(image), 'with dimesions:', image.shape)
plt.imshow(image)  #call as plt.imshow(gray, cmap='gray') to show a grayscaled image


# **Some OpenCV functions (beyond those introduced in the lesson) that might be useful for this project are:**
# 
# `cv2.inRange()` for color selection  
# `cv2.fillPoly()` for regions selection  
# `cv2.line()` to draw lines on an image given endpoints  
# `cv2.addWeighted()` to coadd / overlay two images
# `cv2.cvtColor()` to grayscale or change color
# `cv2.imwrite()` to output images to file  
# `cv2.bitwise_and()` to apply a mask to an image
# 
# **Check out the OpenCV documentation to learn about these and discover even more awesome functionality!**

# Below are some helper functions to help get you started. They should look familiar from the lesson!

# In[5]:

import math

def grayscale(img):
    """Applies the Grayscale transform
    This will return an image with only one color channel
    but NOTE: to see the returned image as grayscale
    you should call plt.imshow(gray, cmap='gray')"""
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
def canny(img, low_threshold, high_threshold):
    """Applies the Canny transform"""
    return cv2.Canny(img, low_threshold, high_threshold)

def gaussian_blur(img, kernel_size):
    """Applies a Gaussian Noise kernel"""
    return cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)

def region_of_interest(img, vertices):
    """
    Applies an image mask.
    
    Only keeps the region of the image defined by the polygon
    formed from `vertices`. The rest of the image is set to black.
    """
    #defining a blank mask to start with
    mask = np.zeros_like(img)   
    
    #defining a 3 channel or 1 channel color to fill the mask with depending on the input image
    if len(img.shape) > 2:
        channel_count = img.shape[2]  # i.e. 3 or 4 depending on your image
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255
        
    #filling pixels inside the polygon defined by "vertices" with the fill color    
    cv2.fillPoly(mask, vertices, ignore_mask_color)
    
    #returning the image only where mask pixels are nonzero
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image


def draw_lines(img, lines, color=[255, 0, 0], thickness=2):
    """
    NOTE: this is the function you might want to use as a starting point once you want to 
    average/extrapolate the line segments you detect to map out the full
    extent of the lane (going from the result shown in raw-lines-example.mp4
    to that shown in P1_example.mp4).  
    
    Think about things like separating line segments by their 
    slope ((y2-y1)/(x2-x1)) to decide which segments are part of the left
    line vs. the right line.  Then, you can average the position of each of 
    the lines and extrapolate to the top and bottom of the lane.
    
    This function draws `lines` with `color` and `thickness`.    
    Lines are drawn on the image inplace (mutates the image).
    If you want to make the lines semi-transparent, think about combining
    this function with the weighted_img() function below
    """
    for line in lines:
        for x1,y1,x2,y2 in line:
            cv2.line(img, (x1, y1), (x2, y2), color, thickness)

def hough_lines(img, rho, theta, threshold, min_line_len, max_line_gap):
    """
    `img` should be the output of a Canny transform.
        
    Returns an image with hough lines drawn.
    """
    lines = cv2.HoughLinesP(img, rho, theta, threshold, np.array([]), minLineLength=min_line_len, maxLineGap=max_line_gap)
    line_img = np.zeros((*img.shape, 3), dtype=np.uint8)
    draw_lines(line_img, lines)
    return line_img

# Python 3 has support for cool math symbols.

def weighted_img(img, initial_img, α=0.8, β=1., λ=0.):
    """
    `img` is the output of the hough_lines(), An image with lines drawn on it.
    Should be a blank image (all black) with lines drawn on it.
    
    `initial_img` should be the image before any processing.
    
    The result image is computed as follows:
    
    initial_img * α + img * β + λ
    NOTE: initial_img and img must be the same shape!
    """
    return cv2.addWeighted(initial_img, α, img, β, λ)


# ## Test on Images
# 
# Now you should build your pipeline to work on the images in the directory "test_images"  
# **You should make sure your pipeline works well on these images before you try the videos.**

# In[6]:

import os
os.listdir("test_images/")


# run your solution on all test_images and make copies into the test_images directory).

# ## Test on Videos
# 
# You know what's cooler than drawing lanes over images? Drawing lanes over video!
# 
# We can test our solution on two provided videos:
# 
# `solidWhiteRight.mp4`
# 
# `solidYellowLeft.mp4`

# In[7]:

# Import everything needed to edit/save/watch video clips
from moviepy.editor import VideoFileClip
from IPython.display import HTML


# In[11]:

def process_image(image):
    # NOTE: The output you return should be a color image (3 channel) for processing video below
    # you should return the final output (image with lines are drawn on lanes)
    result = grayscale(image)
    result = gaussian_blur(result, 5)
    result = canny(result, 50, 150)
    regions = np.array([[
        [0, image.shape[0]], # left bot
        [image.shape[1] * 0.45, image.shape[0] * 0.65],
        [image.shape[1] * 0.55, image.shape[0] * 0.65],
        [image.shape[1], image.shape[0]] # right bot
    ]], dtype=np.int32)
    result = region_of_interest(result, regions)
    plt.imshow(result, cmap='gray')
    
    rho = 2
    theta = np.pi / 180 * 1
    threshold = 10
    min_line_len = 20
    max_line_gap = 10
    img = result
    lines = cv2.HoughLinesP(img, rho, theta, threshold, np.array([]), minLineLength=min_line_len, maxLineGap=max_line_gap)
    lines = process_lines(lines, image)
    line_img = np.zeros((*img.shape, 3), dtype=np.uint8)
    draw_lines(line_img, lines, thickness = 10)
    result = weighted_img(line_img, image, β=0.5)
    
    return result

def process_lines(lines, image):
    '''
    Valid line should intersect the bottem edge, and slope should be in certain range
    Pick average of intersection and slope
    Only left 2 lines: left line has negtive slope, right line has positive slope
    '''
    HEIGHT_R = 0.65 # height ratio for the final line
    results = [{"slope":0.00001, "x":0, "weight": 0} for i in range(2)];
    f_lines = []
    for line in lines:
        for x1,y1,x2,y2 in line:
            if x1 == x2 : continue
            slope = (y1 - y2) * 1.0/ (x1 - x2)
            # line: (y-y2)*(x1-x2) = (x-x2)*(y1-y2), where y = image.shape[0]
            intersect_x = (image.shape[0] - y2) / slope + x2
            if intersect_x < 0 or intersect_x >= image.shape[1]:
                # ignore invalid line
                continue
            if slope > -0.85 and slope < -0.5:
                index = 0
            elif slope > 0.5 and slope < 0.85:
                index = 1
            else:
                continue
            one_r = results[index]
            
            # Collect weighted average, where weight is the length^2 of the line
            weight = (x1 - x2) * (x1 - x2) + (y1 - y2) * (y1 - y2)
            one_r["slope"] += slope * weight
            one_r["x"] += intersect_x * weight
            one_r["weight"] += weight

            
    for one_r in results:
        # draw line up to position height * HEIGHT_R
        if one_r["weight"] == 0:
            f_lines.append([[0,0,0,0]])
            continue
        one_r["x"] /= float(one_r["weight"])
        one_r["slope"] /= float(one_r["weight"])

        point0 = (int(one_r["x"]), image.shape[0])
        # line: y - y0 = slope * (x - x0), where x0,y0 = point0
        point1 = (int(image.shape[0] * (HEIGHT_R - 1) / one_r["slope"] + one_r["x"]), int(image.shape[0] * HEIGHT_R))
        f_lines.append([point0 + point1])
    return f_lines

#import glob
#for efile in glob.glob("test_images/*.jpg"): 
#    image = mpimg.imread(efile)
#    a = process_image(image)
#plt.imshow(process_image(image), cmap="gray")
#plt.imshow(image, cmap="gray")



# Let's try the one with the solid white lane on the right first ...

# In[12]:

white_output = 'white.mp4'
clip1 = VideoFileClip("solidWhiteRight.mp4")
white_clip = clip1.fl_image(process_image) #NOTE: this function expects color images!!
get_ipython().magic('time white_clip.write_videofile(white_output, audio=False)')


# Play the video inline, or if you prefer find the video in your filesystem (should be in the same directory) and play it in your video player of choice.

# In[13]:

HTML("""
<video width="960" height="540" controls>
  <source src="{0}">
</video>
""".format(white_output))


# **At this point, if you were successful you probably have the Hough line segments drawn onto the road, but what about identifying the full extent of the lane and marking it clearly as in the example video (P1_example.mp4)?  Think about defining a line to run the full length of the visible lane based on the line segments you identified with the Hough Transform.  Modify your draw_lines function accordingly and try re-running your pipeline.**

# Now for the one with the solid yellow lane on the left. This one's more tricky!

# In[11]:

yellow_output = 'yellow.mp4'
clip2 = VideoFileClip('solidYellowLeft.mp4')
yellow_clip = clip2.fl_image(process_image)
get_ipython().magic('time yellow_clip.write_videofile(yellow_output, audio=False)')


# In[13]:

HTML("""
<video width="960" height="540" controls>
  <source src="{0}">
</video>
""".format(yellow_output))


# ## Reflections
# 
# Congratulations on finding the lane lines!  As the final step in this project, we would like you to share your thoughts on your lane finding pipeline... specifically, how could you imagine making your algorithm better / more robust?  Where will your current algorithm be likely to fail?
# 
# Please add your thoughts below,  and if you're up for making your pipeline more robust, be sure to scroll down and check out the optional challenge video below!
# 

# In[6]:

# Reflection:
# 1. Current pipeline is:
#  - Convert image to gray
#  - Use gaussian to blur the image and extract the edges
#  - Apply the mask for the middle bottom area that we are interested in
#  - Use Hough transform to extract lines
#  - Categorize all lines into: left line group, right line group and non-valid lines
#  - For left/right line group, calculate the weighted-average of the bottem intersection 
#    point and the slope, and create 2 lines from them.
#  - Apply 2 lines on top of original image

# 2. This algorithm only works well on the bottom half of the image has clear lane mark. 

# 3. To make it more robust, there are several thoughts:
# - We can treat all the lines from Hough transform as points and find 2 lines of best fit. 
# - We can also use the curve to fit instead of straight line
# - This algorithm only consider to extract info from current frame. Because the real world
#   change is always continuous, we can alwasy consider the result from previous frame(s). 
#   The current frame's result shouldn't change too much(like 45 degree to 80 degree).


# ## Submission
# 
# If you're satisfied with your video outputs it's time to submit!  Submit this ipython notebook for review.
# 

# ## Optional Challenge
# 
# Try your lane finding pipeline on the video below.  Does it still work?  Can you figure out a way to make it more robust?  If you're up for the challenge, modify your pipeline so it works with this video and submit it along with the rest of your project!

# In[14]:

challenge_output = 'extra.mp4'
clip2 = VideoFileClip('challenge.mp4')
challenge_clip = clip2.fl_image(process_image)
get_ipython().magic('time challenge_clip.write_videofile(challenge_output, audio=False)')


# In[15]:

HTML("""
<video width="960" height="540" controls>
  <source src="{0}">
</video>
""".format(challenge_output))


# In[2]:




# In[ ]:



