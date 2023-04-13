import time
import scipy.io
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

from cv2 import resize, INTER_CUBIC
from matplotlib.patches import Circle

from ex1_student_solution import Solution

def load_data(is_perfect_matches=False):
    # Read the data:
    src_img = mpimg.imread('src_test.jpg')
    dst_img = mpimg.imread('dst_test.jpg')
    # matching points and some outliers
    matches = scipy.io.loadmat('matches_test')
    match_p_dst = matches['match_p_dst'].astype(float)
    match_p_src = matches['match_p_src'].astype(float)
    return src_img, dst_img, match_p_src, match_p_dst

# loading data with perfect matches
src_img, dst_img, match_src, match_dst = load_data(False)

fig = plt.figure(figsize=(10, 5))

Image1 = src_img
Image2 = dst_img



fig.add_subplot(1, 2, 1)

plt.imshow(Image1)
print(match_src.shape)
plt.scatter(match_src[0,:],match_src[1,:], facecolors='none', edgecolors='r', label='Matches', s=30)
plt.axis('off')
plt.title("src")
plt.legend()

fig.add_subplot(1, 2, 2)
  
plt.imshow(Image2)
plt.scatter(match_dst[0,:],match_dst[1,:], facecolors='none', edgecolors='r', label='Matches', s=30)
plt.axis('off')
plt.title("dst")
plt.legend()

plt.show()
print('ok')