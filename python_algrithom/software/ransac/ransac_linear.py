#-*- coding: utf-8 -*-
"""
Create on 2019/9/14

@Author: xhj
"""


import time
import numpy as np
from ransac import Ransac


class RansacLinear(Ransac):

    def __init__(self, samples=None):

        Ransac.__init__(self, samples)


    def wrap(self, samples):
        """
        Expand the dimensions of each sample
        """

        rows, cols = samples.shape[:2]
        wraped_samples = np.ones((rows, cols + 1))
        wraped_samples[:, :cols] = samples
        return wraped_samples


    def computeModel(self, sample_num=3, stop_at_goal=True):
        """
        Compute the fitting model with ransac algorithm.
        Args:
            sample_num: the number of samples collect in every iteration
            stop_at_goal: bool, stop iteration when the number of inliers is greater
                          than threshold
        """

        max_inliers_num = 0
        skip_sample_num = 0
        idx = 0
        max_iterations = self.max_iterations

        # generate the index of sampling in every iteration
        random_sample_array = list(np.random.randint(0, self.global_samples_num, \
                                                     (self.max_iterations, sample_num)))

        def is_inlier(coeff, samples, threshold):
            return np.abs(coeff.dot(self.wrap(samples).T)) < threshold

        while idx < max_iterations:      # iterative solution
            
            sample_index = random_sample_array[idx]
            if len(set(sample_index)) != len(sample_index):
                random_sample_array.append(np.random.randint(0, self.global_samples_num, sample_num))
                max_iterations += 1
                skip_sample_num += 1
                if skip_sample_num > self.global_samples_num / 10:
                    return self.coeff
                continue

            coeff_calc = lambda samples: np.linalg.svd(self.wrap(samples))[-1][-1, :]
            coeff = coeff_calc(self.samples[sample_index])
            self.inliers_mask = is_inlier(coeff, self.samples, self.distance_thresh).astype(np.int32)
            inliers_num = np.sum(self.inliers_mask, axis=0)

            if inliers_num > max_inliers_num:
                max_inliers_num = inliers_num
                self.coeff = coeff

                if stop_at_goal and max_inliers_num > self.inlier_thresh:
                    return self.coeff

            idx += 1
        
        return self.coeff


if __name__ == '__main__':

    def test_2d():

        import matplotlib
        import matplotlib.pyplot as plt

        n = 500
        max_iterations = 100
        goal_inliers = n * 0.4

        xys = np.random.random((n, 2)) * 100
        xys[:200, 1:] = xys[:200, :1]

        plt.scatter(xys.T[0], xys.T[1])

        # RANSAC
        start = time.time()
        ransac = RansacLinear(xys)
        ransac.set_inlier_thresh(goal_inliers)
        ransac.set_distance_thresh(0.1)
        ransac.set_max_iterations(200)
        a, b, c = ransac.computeModel(3)[:]
        
        plt.plot([0, 100], [-c/b, -(c + 100 * a) / b], color=(0, 1, 0))
        print("time used: ", time.time() - start)
        plt.show()


    def test_3d():

        import matplotlib
        import matplotlib.pyplot as plt
        from mpl_toolkits import mplot3d

        fig = plt.figure()
        ax = mplot3d.Axes3D(fig)

        def plot_plane(a, b, c, d):
            xx, yy = np.mgrid[:10, :10]
            return xx, yy, (-d - a * xx - b * yy) / c

        n = 500
        max_iterations = 100
        goal_inliers = n * 0.3

        # test data
        xyzs = np.random.random((n, 3)) * 10
        xyzs[:250, 2:] = xyzs[:250, :1]

        ax.scatter3D(xyzs.T[0], xyzs.T[1], xyzs.T[2])

        # RANSAC
        start = time.time()
        ransac = RansacLinear(xyzs)
        ransac.set_inlier_thresh(goal_inliers)
        ransac.set_distance_thresh(0.1)
        ransac.set_max_iterations(200)
        a, b, c, d = ransac.computeModel(3)[:]
        print("time used: ", time.time() - start)
        xx, yy, zz = plot_plane(a, b, c, d)
        ax.plot_surface(xx, yy, zz, color=(0, 1, 0, 0.5))

        plt.show()

    test_2d()
