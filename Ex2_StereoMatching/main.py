import time
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

from solution import Solution


COST1 = 0.5
COST2 = 3.0
WIN_SIZE = 3
DISPARITY_RANGE = 20
##########################################################
# Don't forget to fill in your IDs!!!
# students' IDs:
ID1 = '301613501'
ID2 = '305207946'
##########################################################


def tic():
    return time.time()


def toc(t):
    return float(tic()) - float(t)


def forward_map(left_image, labels):
    labels -= DISPARITY_RANGE
    mapped = np.zeros_like(left_image)
    for row in range(left_image.shape[0]):
        cols = range(left_image.shape[1])
        mapped[row,
               np.clip(cols - labels[row, ...], 0, left_image.shape[1] - 1),
               ...] = left_image[row, cols, ...]
    return mapped


def load_data(is_your_data=False):
    # Read the data:
    if is_your_data:
        left_image = mpimg.imread('my_image_left.png')
        right_image = mpimg.imread('my_image_right.png')
    else:
        left_image = mpimg.imread('image_left.png')
        right_image = mpimg.imread('image_right.png')
    return left_image, right_image


def main():
    COST1 = 0.5
    COST2 = 3.0
    WIN_SIZE = 3
    DISPARITY_RANGE = 20
    left_image, right_image = load_data()
    solution = Solution()
    # Compute Sum-Square-Diff distance
    tt = tic()
    ssdd = solution.ssd_distance(left_image.astype(np.float64),
                                 right_image.astype(np.float64),
                                 win_size=WIN_SIZE,
                                 dsp_range=DISPARITY_RANGE)
    print(f"SSDD calculation done in {toc(tt):.4f}[seconds]")

    # Construct naive disparity image
    tt = tic()
    label_map = solution.naive_labeling(ssdd)
    print(f"Naive labeling done in {toc(tt):.4f}[seconds]")

    # plot the left image and the estimated depth
    fig = plt.figure()
    plt.subplot(1, 2, 1)
    plt.imshow(left_image)
    plt.subplot(1, 2, 2)
    plt.imshow(label_map)
    plt.colorbar(shrink=0.4)
    plt.title('Naive Depth')
    plt.tight_layout()
    # plt.savefig('Figure 1 - Naive Depth.png', format='PNG')

    # Compute forward map of the left image to the right image.
    mapped_image_naive_dp = forward_map(left_image, labels=label_map)
    # plot left image, forward map image and right image
    plt.figure()
    plt.subplot(1, 3, 1)
    plt.imshow(left_image)
    plt.title('Source Image')
    plt.subplot(1, 3, 2)
    plt.imshow(mapped_image_naive_dp)
    plt.title('Naive Forward map - DP')
    plt.subplot(1, 3, 3)
    plt.imshow(right_image)
    plt.title('Right Image')
    plt.tight_layout()
    #plt.savefig('Figure 1_1 - Left_ForwardMap_Right.png', format='PNG')

    # Smooth disparity image - Dynamic Programming
    tt = tic()
    label_smooth_dp = solution.dp_labeling(ssdd, COST1, COST2)
    print(f"Dynamic Programming done in {toc(tt):.4f}[seconds]")

    # plot the left image and the estimated depth
    plt.figure()
    plt.subplot(1, 2, 1)
    plt.imshow(left_image)
    plt.title('Source Image')
    plt.subplot(1, 2, 2)
    plt.imshow(label_smooth_dp)
    plt.colorbar(shrink=0.4)
    plt.title('Smooth Depth - DP')
    plt.tight_layout()
    #plt.savefig('Figure 2 - Smooth Depth - DP.png', format='PNG')

    # Compute forward map of the left image to the right image.
    mapped_image_smooth_dp = forward_map(left_image, labels=label_smooth_dp)
    # plot left image, forward map image and right image
    plt.figure()
    plt.subplot(1, 3, 1)
    plt.imshow(left_image)
    plt.title('Source Image')
    plt.subplot(1, 3, 2)
    plt.imshow(mapped_image_smooth_dp)
    plt.title('Smooth Forward map - DP')
    plt.subplot(1, 3, 3)
    plt.imshow(right_image)
    plt.title('Right Image')
    plt.tight_layout()
    #plt.savefig('Figure 3 - Left_ForwardMap_Right.png', format='PNG')

    # Generate a dictionary which maps each direction to a label map:
    tt = tic()
    direction_to_vote = solution.dp_labeling_per_direction(ssdd, COST1, COST2)
    print(f"Dynamic programming in all directions done in {toc(tt):.4f}"
          f"[seconds]")

    # Plot all directions as well as the image, in the center of the plot:
    plt.figure()
    for i in range(1, 1 + 9):
        plt.subplot(3, 3, i)
        if i < 5:
            plt.imshow(direction_to_vote[i])
            plt.title(f'Direction {i}')
        elif i == 5:
            plt.imshow(left_image)
            plt.title(f'Left Image')
        else:
            plt.imshow(direction_to_vote[i - 1])
            plt.title(f'Direction {i - 1}')
    plt.tight_layout()
    #plt.savefig('Figure 4 - All Directions.png', format='PNG')

    # Smooth disparity image - Semi-Global Mapping
    tt = tic()
    label_smooth_sgm = solution.sgm_labeling(ssdd, COST1, COST2)
    print(f"SGM done in {toc(tt):.4f}[seconds]")

    # Plot Semi-Global Mapping result:
    plt.figure()
    plt.subplot(1, 2, 1)
    plt.imshow(left_image)
    plt.title('Source Image')
    plt.subplot(1, 2, 2)
    plt.imshow(label_smooth_sgm)
    plt.colorbar(shrink=0.4)
    plt.title('Smooth Depth - SGM')
    plt.tight_layout()
    #plt.savefig('Figure 5 - Smooth Depth - SGM.png', format='PNG')

    # Plot the forward map based on the Semi-Global Mapping result:
    mapped_image_smooth_sgm = forward_map(left_image, labels=label_smooth_sgm)
    plt.figure()
    plt.subplot(1, 3, 1)
    plt.imshow(left_image)
    plt.title('Source Image')
    plt.subplot(1, 3, 2)
    plt.imshow(mapped_image_smooth_sgm)
    plt.title('Smooth Forward map - SGM')
    plt.subplot(1, 3, 3)
    plt.imshow(right_image)
    plt.title('Right Image')
    plt.tight_layout()
    #plt.savefig('Figure 6 - Left_SGM_Right.png', format='PNG')
    
    ###########################################################################
    ################################### BONUS #################################
    ###########################################################################
    # # Compute Sum-Square-Diff distance
    # tt = tic()
    # ssdd = solution.abs_distance(left_image.astype(np.float64),
    #                              right_image.astype(np.float64),
    #                              win_size=WIN_SIZE,
    #                              dsp_range=DISPARITY_RANGE)
    # print(f"Bonus SabsD calculation done in {toc(tt):.4f}[seconds]")

    # # Construct naive disparity image
    # tt = tic()
    # label_map = solution.naive_labeling(ssdd)
    # print(f"Bonus Naive labeling done in {toc(tt):.4f}[seconds]")

    # # plot the left image and the estimated depth
    # fig = plt.figure()
    # plt.subplot(1, 2, 1)
    # plt.imshow(left_image)
    # plt.subplot(1, 2, 2)
    # plt.imshow(label_map)
    # plt.colorbar(shrink=0.4)
    # plt.title('Bonus - Naive Depth')
    # plt.tight_layout()
    # #plt.savefig('Figure 1 Bonus - Naive Depth.png', format='PNG')

    # # Smooth disparity image - Dynamic Programming
    # tt = tic()
    # label_smooth_dp = solution.dp_labeling(ssdd, COST1, COST2)
    # print(f"Bonus Dynamic Programming done in {toc(tt):.4f}[seconds]")

    # # plot the left image and the estimated depth
    # plt.figure()
    # plt.subplot(1, 2, 1)
    # plt.imshow(left_image)
    # plt.title('Source Image')
    # plt.subplot(1, 2, 2)
    # plt.imshow(label_smooth_dp)
    # plt.colorbar(shrink=0.4)
    # plt.title('Bonus Smooth Depth - DP')
    # plt.tight_layout()
    # #plt.savefig('Figure 2 Bonus - Smooth Depth - DP.png', format='PNG')

    # # Smooth disparity image - Semi-Global Mapping
    # tt = tic()
    # label_smooth_sgm = solution.sgm_labeling(ssdd, COST1, COST2)
    # print(f"Bonus SGM done in {toc(tt):.4f}[seconds]")

    # # Plot Semi-Global Mapping result:
    # plt.figure()
    # plt.subplot(1, 2, 1)
    # plt.imshow(left_image)
    # plt.title('Source Image')
    # plt.subplot(1, 2, 2)
    # plt.imshow(label_smooth_sgm)
    # plt.colorbar(shrink=0.4)
    # plt.title('Bonus Smooth Depth - SGM')
    # plt.tight_layout()
    # #plt.savefig('Figure 5 Bonus - Smooth Depth - SGM.png', format='PNG')

    ###########################################################################
    ########################### YOUR IMAGE PLAYGROUND #########################
    ###########################################################################
    COST1 = 0.5           # YOU MAY CHANGE THIS
    COST2 = 2.0           # YOU MAY CHANGE THIS
    WIN_SIZE = 3          # YOU MAY CHANGE THIS
    DISPARITY_RANGE = 40  # YOU MAY CHANGE THIS

    your_left_image, your_right_image = load_data(is_your_data=True)
    solution = Solution()
    # Compute Sum-Square-Diff distance
    tt = tic()
    your_ssdd = solution.ssd_distance(your_left_image.astype(np.float64),
                                      your_right_image.astype(np.float64),
                                      win_size=WIN_SIZE,
                                      dsp_range=DISPARITY_RANGE)
    print(f"SSDD calculation on your image took: {toc(tt):.4f}[seconds]")

    # plot all directions as well as the image, in the center of the plot
    tt = tic()
    your_direction_to_vote = solution.dp_labeling_per_direction(your_ssdd,
                                                                COST1,
                                                                COST2)
    print(f"Dynamic programming in all directions took: {toc(tt):.4f}[seconds]")
    # Plot all directions as well as the image, in the center of the plot:
    plt.figure()
    for i in range(1, 1 + 9):
        plt.subplot(3, 3, i)
        if i < 5:
            plt.imshow(your_direction_to_vote[i])
            plt.title(f'Direction {i}')
        elif i == 5:
            plt.imshow(your_left_image)
            plt.title(f'Your Left Image')
        else:
            plt.imshow(your_direction_to_vote[i - 1])
            plt.title(f'Direction {i - 1}')
    plt.tight_layout()
    #plt.savefig('Figure 7 - Your All Directions.png', format='PNG')

    # Smooth disparity image - Semi-Global Mapping
    tt = tic()
    your_label_smooth_sgm = solution.sgm_labeling(your_ssdd, COST1, COST2)
    print(f"SGM on your image done in {toc(tt):.4f}[seconds]")
    plt.figure()
    plt.subplot(1, 2, 1)
    plt.imshow(your_left_image)
    plt.title('Your Source Image')
    plt.subplot(1, 2, 2)
    plt.imshow(your_label_smooth_sgm)
    plt.colorbar(shrink=0.4)
    plt.title('Your Smooth Depth - SGM')
    plt.tight_layout()
    #plt.savefig('Figure 8 - Your Smooth Depth - SGM.png', format='PNG')
    

    # Plot the forward map based on the Semi-Global Mapping result:
    your_mapped_image_smooth_sgm = forward_map(your_left_image,
                                               labels=your_label_smooth_sgm)
    plt.figure()
    plt.subplot(1, 3, 1)
    plt.imshow(your_left_image)
    plt.title('Your Source Image')
    plt.subplot(1, 3, 2)
    plt.imshow(your_mapped_image_smooth_sgm)
    plt.title('Your Smooth Forward map - SGM')
    plt.subplot(1, 3, 3)
    plt.imshow(your_right_image)
    plt.title('Your Right Image')
    plt.tight_layout()
    #plt.savefig('Figure 9 - Your Left_SGM_Right.png', format='PNG')

    plt.show()


if __name__ == "__main__":
    main()
