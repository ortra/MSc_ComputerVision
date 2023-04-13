import time
import scipy.io
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

from cv2 import resize, INTER_CUBIC
from matplotlib.patches import Circle

from ex1_student_solution import Solution

def load_data(is_perfect_matches=True):
    # Read the data:
    src_img = mpimg.imread('images/src.jpg')
    dst_img = mpimg.imread('images/dst.jpg')
    if is_perfect_matches:
        # loading perfect matches
        matches = scipy.io.loadmat('data/matches_perfect')
    else:
        # matching points and some outliers
        matches = scipy.io.loadmat('data/matches')
    match_p_dst = matches['match_p_dst'].astype(float)
    match_p_src = matches['match_p_src'].astype(float)
    return src_img, dst_img, match_p_src, match_p_dst

# loading data with perfect matches
src_img, dst_img, match_p_src, match_p_dst = load_data()
_, _, match_src, match_dst = load_data(False)

fig = plt.figure(figsize=(10, 5))

Image1 = src_img
Image2 = dst_img



fig.add_subplot(1, 2, 1)

plt.imshow(Image1)
print(match_p_src.shape)
plt.scatter(match_p_src[0,:],match_p_src[1,:], facecolors='none', edgecolors='g', label='Perfect Match', s=40)
plt.scatter(match_src[0,:],match_src[1,:], facecolors='r', edgecolors='none', marker='X',label='Match', s=10)
plt.axis('off')
plt.title("src")
plt.legend()

fig.add_subplot(1, 2, 2)
  
# showing image
plt.imshow(Image2)
plt.scatter(match_p_dst[0,:],match_p_dst[1,:], facecolors='none', edgecolors='g', label='Perfect Match', s=40)
plt.scatter(match_dst[0,:],match_dst[1,:], facecolors='r', edgecolors='none', marker='X',label='Match', s=10)
plt.axis('off')
plt.title("dst")
plt.legend()

plt.show()
print('ok')