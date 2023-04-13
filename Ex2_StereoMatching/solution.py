"""Stereo matching."""
import numpy as np
from scipy.signal import convolve2d
from scipy.signal.ltisys import zeros_like


class Solution:
    def __init__(self):
        pass
    
    @staticmethod
    def abs_distance(left_image: np.ndarray,
                     right_image: np.ndarray,
                     win_size: int,
                     dsp_range: int) -> np.ndarray:
        """Compute the SabsD distances tensor.

        Args:
            left_image: Left image of shape: HxWx3, and type np.double64.
            right_image: Right image of shape: HxWx3, and type np.double64.
            win_size: Window size odd integer.
            dsp_range: Half of the disparity range. The actual range is
            -dsp_range, -dsp_range + 1, ..., 0, 1, ..., dsp_range.

        Returns:
            A tensor of the sum of absolute differences for every pixel in a
            window of size win_size X win_size, for the 2*dsp_range + 1
            possible disparity values. The tensor shape should be:
            HxWx(2*dsp_range+1).
        """
        num_of_rows, num_of_cols = left_image.shape[0], left_image.shape[1]
        disparity_values = range(-dsp_range, dsp_range+1)
        sabsd_tensor = np.zeros((num_of_rows,
                                num_of_cols,
                                len(disparity_values)))
        """INSERT YOUR CODE HERE"""
        k = win_size // 2
        
        channels = left_image.shape[2]
        
        for i, d in enumerate(disparity_values):
          left_pad = np.pad(left_image, ((k,k), (abs(d)+k,abs(d)+k), (0,0)))
          right_pad = np.pad(right_image, ((k,k), (abs(d)+k,abs(d)+k), (0,0)))
          right_new = zeros_like(right_pad)
          crop = k+abs(d)+1 # the shift for crop
          right_new[:,crop:(crop+num_of_cols),:] = right_pad[:,(crop+d):(crop+d+num_of_cols),:]

          # Step2: Create window for each image, and substract the the values
          kernel = np.ones((win_size,win_size))

          dist = left_pad-right_new
          absd = np.abs(dist) # squared distance
          sabsd = np.zeros_like(absd[:,:,0])
          for c in range(channels): # RGB Image
            sabsd[:,:] += convolve2d(absd[:,:,c],kernel, 'same')
          
          sabsd_tensor[:,:,i] = sabsd[1:(1+num_of_rows),(k+abs(d)):((k+abs(d))+num_of_cols)]

        sabsd_tensor -= sabsd_tensor.min()
        sabsd_tensor /= sabsd_tensor.max()
        sabsd_tensor *= 255.0

        return sabsd_tensor

    @staticmethod
    def ssd_distance(left_image: np.ndarray,
                     right_image: np.ndarray,
                     win_size: int,
                     dsp_range: int) -> np.ndarray:
        """Compute the SSDD distances tensor.

        Args:
            left_image: Left image of shape: HxWx3, and type np.double64.
            right_image: Right image of shape: HxWx3, and type np.double64.
            win_size: Window size odd integer.
            dsp_range: Half of the disparity range. The actual range is
            -dsp_range, -dsp_range + 1, ..., 0, 1, ..., dsp_range.

        Returns:
            A tensor of the sum of squared differences for every pixel in a
            window of size win_size X win_size, for the 2*dsp_range + 1
            possible disparity values. The tensor shape should be:
            HxWx(2*dsp_range+1).
        """
        num_of_rows, num_of_cols = left_image.shape[0], left_image.shape[1]
        disparity_values = range(-dsp_range, dsp_range+1)
        ssdd_tensor = np.zeros((num_of_rows,
                                num_of_cols,
                                len(disparity_values)))
        """INSERT YOUR CODE HERE"""
        k = win_size // 2
        
        channels = left_image.shape[2]
        
        for i, d in enumerate(disparity_values):
          left_pad = np.pad(left_image, ((k,k), (abs(d)+k,abs(d)+k), (0,0)))
          right_pad = np.pad(right_image, ((k,k), (abs(d)+k,abs(d)+k), (0,0)))
          right_new = zeros_like(right_pad)
          crop = k+abs(d)+1 # the shift for crop
          right_new[:,crop:(crop+num_of_cols),:] = right_pad[:,(crop+d):(crop+d+num_of_cols),:]

          # Step2: Create window for each image, and substract the the values
          kernel = np.ones((win_size,win_size))

          dist = left_pad-right_new
          sd = (np.power(dist,2)) # squared distance
          ssd = np.zeros_like(sd[:,:,0])
          for c in range(channels): # RGB Image
            ssd[:,:] += convolve2d(sd[:,:,c],kernel, 'same')
          
          ssdd_tensor[:,:,i] = ssd[1:(1+num_of_rows),(k+abs(d)):((k+abs(d))+num_of_cols)]

        ssdd_tensor -= ssdd_tensor.min()
        ssdd_tensor /= ssdd_tensor.max()
        ssdd_tensor *= 255.0

        return ssdd_tensor

    @staticmethod
    def naive_labeling(ssdd_tensor: np.ndarray) -> np.ndarray:
        """Estimate a naive depth estimation from the SSDD tensor.

        Args:
            ssdd_tensor: A tensor of the sum of squared differences for every
            pixel in a window of size win_size X win_size, for the
            2*dsp_range + 1 possible disparity values.

        Evaluate the labels in a naive approach. Each value in the
        result tensor should contain the disparity matching minimal ssd (sum of
        squared difference).

        Returns:
            Naive labels HxW matrix.
        """
        # you can erase the label_no_smooth initialization.
        label_no_smooth = np.zeros((ssdd_tensor.shape[0], ssdd_tensor.shape[1]))
        """INSERT YOUR CODE HERE"""
        label_no_smooth = np.argmin(ssdd_tensor, axis=2)
        return label_no_smooth

    @staticmethod
    def dp_grade_slice(c_slice: np.ndarray, p1: float, p2: float) -> np.ndarray:
        """Calculate the scores matrix for slice c_slice.

        Calculate the scores slice which for each column and disparity value
        states the score of the best route. The scores slice is of shape:
        (2*dsp_range + 1)xW.

        Args:
            c_slice: A slice of the ssdd tensor.
            p1: penalty for taking disparity value with 1 offset.
            p2: penalty for taking disparity value more than 2 offset.
        Returns:
            Scores slice which for each column and disparity value states the
            score of the best route.
        """
        num_labels, num_of_cols = c_slice.shape[0], c_slice.shape[1]
        l_slice = np.zeros((num_labels, num_of_cols))
        """INSERT YOUR CODE HERE"""
                # Creating a penalty matrix
        P = np.zeros((num_labels,num_labels)) 
        P[np.triu_indices(num_labels,k=1)] = p1
        P[np.triu_indices(num_labels,k=2)] = p2
        P = P + P.T

        l_slice[:,0] = c_slice[:,0] # initialialize the first column

        for col in range(1,num_of_cols):
          # Duplicate the (col-1) column to num_labels vectors
          L = np.tile(l_slice[:,col-1], (num_labels, 1))
          # For each pixel add the relevant penalty
          L_penalty = L + P
          # Taking the minimum value from each row 
          m_slice = np.min(L_penalty, axis=1) # vector of num_labels length 
          l_slice[:,col] = c_slice[:,col] + m_slice - np.min(l_slice[:,col-1])
        return l_slice

    def dp_labeling(self,
                    ssdd_tensor: np.ndarray,
                    p1: float,
                    p2: float) -> np.ndarray:
        """Estimate a depth map using Dynamic Programming.

        (1) Call dp_grade_slice on each row slice of the ssdd tensor.
        (2) Store each slice in a corresponding l tensor (of shape as ssdd).
        (3) Finally, for each pixel in l (along each row and column), choose
        the best disparity value. That is the disparity value which
        corresponds to the lowest l value in that pixel.

        Args:
            ssdd_tensor: A tensor of the sum of squared differences for every
            pixel in a window of size win_size X win_size, for the
            2*dsp_range + 1 possible disparity values.
            p1: penalty for taking disparity value with 1 offset.
            p2: penalty for taking disparity value more than 2 offset.
        Returns:
            Dynamic Programming depth estimation matrix of shape HxW.
        """
        l = np.zeros_like(ssdd_tensor)
        """INSERT YOUR CODE HERE"""
        M,N,D = l.shape
        for m in range(M):
          l[m,:,:] = self.dp_grade_slice(ssdd_tensor[m,:,:].T, p1, p2).T
        return self.naive_labeling(l)
    
    def slice_per_direction(self,
                            ssdd_tensor: np.ndarray,
                            direction: int,
                            idx: int) -> np.ndarray  :

      # Create an indices matrix for the ssdd_tensor
      indices_arr = np.arange(ssdd_tensor.size).reshape(ssdd_tensor.shape)

      if direction == 1: # a row at a time
          c_slice = ssdd_tensor[idx, :] 
          c_slice_indices = np.unravel_index(indices_arr[idx,:], ssdd_tensor.shape)
          return c_slice, c_slice_indices 
      
      elif direction == 2: # a diagonal at a time
          c_slice = ssdd_tensor.diagonal(idx).T 
          c_slice_indices = np.unravel_index(indices_arr.diagonal(idx).T, ssdd_tensor.shape)
          return c_slice, c_slice_indices 
      
      elif direction == 3: # a column at a time
          c_slice = ssdd_tensor[:, idx] 
          c_slice_indices = np.unravel_index(indices_arr[:,idx], ssdd_tensor.shape)
          return c_slice, c_slice_indices 
      
      elif direction == 4: # a flipped diagonal (left to right) at a time 
          c_slice = np.fliplr(ssdd_tensor).diagonal(idx).T 
          c_slice_indices = np.unravel_index(np.fliplr(indices_arr).diagonal(idx).T, ssdd_tensor.shape)
          return c_slice, c_slice_indices 
      
      elif direction == 5: # a reversed row at a time
          c_slice = np.fliplr(ssdd_tensor)[idx, :] 
          c_slice_indices = np.unravel_index(np.fliplr(indices_arr)[idx,:], ssdd_tensor.shape)
          return c_slice, c_slice_indices 
      
      elif direction == 6: # # a reveresed diagonal at a time
          c_slice = np.fliplr(np.flipud(ssdd_tensor)).diagonal(idx).T
          c_slice_indices = np.unravel_index(np.fliplr(np.flipud(indices_arr)).diagonal(idx).T, ssdd_tensor.shape)
          return c_slice, c_slice_indices 
      
      elif direction == 7: # a reveresed column at a time
          c_slice = np.flipud(ssdd_tensor)[:, idx]
          c_slice_indices = np.unravel_index(np.flipud(indices_arr)[:,idx], ssdd_tensor.shape)
          return c_slice, c_slice_indices 
      
      elif direction == 8: # a reveresed flipped diagonal (left to right) at a time
          c_slice = np.flipud(ssdd_tensor).diagonal(idx).T
          c_slice_indices = np.unravel_index(np.flipud(indices_arr).diagonal(idx).T, ssdd_tensor.shape)
          return c_slice, c_slice_indices 
        

    def dp_labeling_per_direction(self,
                                  ssdd_tensor: np.ndarray,
                                  p1: float,
                                  p2: float) -> dict:
        """Return a dictionary of directions to a Dynamic Programming
        etimation of depth.

        For each direction in 1, ..., 8, calculate scores tensors
        according to dp_grade_slice and the method which allows you to
        extract slices along each direction.

        You may use helper methods (functions) that you write on your own.
        We found `np.diagonal` to be very helpful to extract diagonal slices.
        `np.unravel_index` might be helpful if you're thinking in MATLAB
        notations: it's the ind2sub equivalent.

        Args:
            ssdd_tensor: A tensor of the sum of squared differences for
            every pixel in a window of size win_size X win_size, for the
            2*dsp_range + 1 possible disparity values.
            p1: penalty for taking disparity value with 1 offset.
            p2: penalty for taking disparity value more than 2 offset.

        Returns:
            Dictionary int->np.ndarray which maps each direction to the
            corresponding dynamic programming estimation of depth based on
            that direction.
        """
        num_of_directions = 8
        l = np.zeros_like(ssdd_tensor)
        direction_to_slice = {}
        """INSERT YOUR CODE HERE"""
        ssdd_rows, ssdd_cols, _ = ssdd_tensor.shape
        direction_range = {}
        direction_range[1] = range(ssdd_rows)                 # r=1
        direction_range[2] = range(-ssdd_rows+1, ssdd_cols)   # r=2
        direction_range[3] = range(ssdd_cols)                 # r=3
        direction_range[4] = range(-ssdd_rows+1, ssdd_cols)   # r=4
        direction_range[5] = np.copy(direction_range[1])      # r=5
        direction_range[6] = np.copy(direction_range[2])      # r=6
        direction_range[7] = np.copy(direction_range[3])      # r=7
        direction_range[8] = np.copy(direction_range[4])      # r=8
        
        for r in range(1, num_of_directions+1):
          for i in direction_range[r]:
            c_slice, idx = self.slice_per_direction(ssdd_tensor,r,i)
            l[idx] = self.dp_grade_slice(c_slice.T,p1,p2).T
          
          direction_to_slice[r] = self.naive_labeling(l)
        return direction_to_slice

    def sgm_labeling(self, ssdd_tensor: np.ndarray, p1: float, p2: float):
        """Estimate the depth map according to the SGM algorithm.

        For each direction in 1, ..., 8, calculate scores tensors
        according to dp_grade_slice and the method which allows you to
        extract slices along each direction.

        You may use helper methods (functions) that you write on your own.
        We found `np.diagonal` to be very helpful to extract diagonal slices.
        `np.unravel_index` might be helpful if you're thinking in MATLAB
        notations: it's the ind2sub equivalent.

        Args:
            ssdd_tensor: A tensor of the sum of squared differences for
            every pixel in a window of size win_size X win_size, for the
            2*dsp_range + 1 possible disparity values.
            p1: penalty for taking disparity value with 1 offset.
            p2: penalty for taking disparity value more than 2 offset.

        Returns:
            Semi-Global Mapping depth estimation matrix of shape HxW.
        """
        num_of_directions = 8
        l = np.zeros_like(ssdd_tensor)
        """INSERT YOUR CODE HERE"""
        ssdd_rows, ssdd_cols, ssdd_chnls = ssdd_tensor.shape
        direction_range = {}
        direction_range[1] = range(ssdd_rows)                 # r=1
        direction_range[2] = range(-ssdd_rows+1, ssdd_cols)   # r=2
        direction_range[3] = range(ssdd_cols)                 # r=3
        direction_range[4] = range(-ssdd_rows+1, ssdd_cols)   # r=4
        direction_range[5] = np.copy(direction_range[1])      # r=5
        direction_range[6] = np.copy(direction_range[2])      # r=6
        direction_range[7] = np.copy(direction_range[3])      # r=7
        direction_range[8] = np.copy(direction_range[4])      # r=8

        l_all_dir = np.zeros((num_of_directions, ssdd_rows, ssdd_cols, ssdd_chnls))

        for r in range(1, num_of_directions+1): # run on 8 directions
          for i in direction_range[r]:
            c_slice, idx = self.slice_per_direction(ssdd_tensor,r,i)
            l_all_dir[r-1][idx] = self.dp_grade_slice(c_slice.T,p1,p2).T
        l = np.mean(l_all_dir, axis=0)                   
        return self.naive_labeling(l)

