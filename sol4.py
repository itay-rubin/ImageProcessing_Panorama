import numpy as np
import os
import matplotlib.pyplot as plt

from scipy.ndimage.morphology import generate_binary_structure
from scipy.ndimage.filters import maximum_filter, convolve
from scipy.ndimage import label, center_of_mass, map_coordinates
import shutil
from imageio import imwrite

import sol4_utils

X_DEV = "x"
Y_DEV = "y"
ROWS = 0
COLS = 1
BLUR_SIZE = 3
K = 0.04
X = 0
Y = 1
Z = 2
PYR_LEVEL = 2
ORIGINAL_IM = 0
SMALLEST_IM = 2
DEF_RAD = 3
SPREAD_RAD = 30
PATCH_SIZE = 7
HOMOGENOUS_COORDS = 3
NO_INLIERS = 0
TRANS_MAT_SIZE = 3
TRANS_ONLY = 1
TRANS_ROTATION = 2
INLIER_TOL = 6
NUM_OF_ITER = 100


def harris_corner_detector(im):
    """
    Detects harris corners.
    Make sure the returned coordinates are x major!!!
    :param im: A 2D array representing an image.
    :return: An array with shape (N,2), where ret[i,:] are the [x,y] coordinates of the ith corner points.
    """
    Ix2, Iy2, Ixy = get_M(im)

    R = Ix2 * Iy2 - np.square(Ixy) - K * np.square(Ix2 + Iy2)

    local_max = non_maximum_suppression(R)
    corners = np.argwhere(local_max)
    return corners[:, [1, 0]]


def get_der(der):
    """
    :param der: String: x or y
    :return: numpy array of the filter
    """
    der_vec = np.array([1, 0, -1])

    if der == X_DEV:
        return np.expand_dims(der_vec, ROWS)

    else:
        return np.expand_dims(der_vec, COLS)


def derivative(im, der_vec):
    """
    Calculates a derivative of an image in a direction according to given filter
    :param im: grayscale image - numpy array of dtype float64
    :param der_vec: numpy array of the derivative filter
    :return: the derivative - numpy array in shape of im
    """
    return convolve(im, der_vec, mode='nearest')


def get_M(im):
    """
    :param im: grayscale image: numpy array of dtype float64
    :return: Ix^2, Iy^2, IxIy : numpy arrays of dtype float64
    """
    x_der = derivative(im, get_der(X_DEV))
    y_der = derivative(im, get_der(Y_DEV))
    Ixy = x_der * y_der

    return sol4_utils.blur_spatial(np.square(x_der), BLUR_SIZE),\
           sol4_utils.blur_spatial(np.square(y_der), BLUR_SIZE),\
           sol4_utils.blur_spatial(Ixy, BLUR_SIZE)


def get_patch(pos, im, desc_rad):
    """
    :param im: grayscale image of dtype float64
    :param pos: tuple of the point to patch according to
    :param k:
    :return: the intensity patch from the image
    """
    pos = pos / np.power(2, PYR_LEVEL)
    x_vec = np.arange(pos[X] - desc_rad, pos[X] + desc_rad + 1)
    y_vec = np.arange(pos[Y] - desc_rad, pos[Y] + desc_rad + 1)

    cols, rows = np.meshgrid(x_vec, y_vec)
    patch = map_coordinates(im, [rows, cols], order=1, prefilter=False)

    if np.linalg.norm(patch - patch.mean()) == 0.0:
        return patch - patch.mean()

    return (patch - patch.mean()) / np.linalg.norm(patch - patch.mean())


def sample_descriptor(im, pos, desc_rad):
    """
    Samples descriptors at the given corners.
    :param im: A 2D array representing an image.
    :param pos: An array with shape (N,2), where pos[i,:] are the [x,y] coordinates of the ith corner point.
    :param desc_rad: "Radius" of descriptors to compute.
    :return: A 3D array with shape (N,K,K) containing the ith descriptor at desc[i,:,:].
    """
    return np.apply_along_axis(get_patch, COLS, pos, im, desc_rad)


def find_features(pyr):
    """
    Detects and extracts feature points from a pyramid.
    :param pyr: Gaussian pyramid of a grayscale image having 3 levels.
    :return: A list containing:
                1) An array with shape (N,2) of [x,y] feature location per row found in the image.
                   These coordinates are provided at the pyramid level pyr[0].
                2) A feature descriptor array with shape (N,K,K)
    """
    pts = spread_out_corners(pyr[ORIGINAL_IM], PATCH_SIZE, PATCH_SIZE, SPREAD_RAD)
    return [pts, sample_descriptor(pyr[SMALLEST_IM], pts, DEF_RAD)]


def match_features(desc1, desc2, min_score):
    """
    Return indices of matching descriptors.
    :param desc1: A feature descriptor array with shape (N1,K,K).
    :param desc2: A feature descriptor array with shape (N2,K,K).
    :param min_score: Minimal match score.
    :return: A list containing:
                1) An array with shape (M,) and dtype int of matching indices in desc1.
                2) An array with shape (M,) and dtype int of matching indices in desc2.
    """
    s = desc1.reshape(desc1.shape[0], -1).dot(desc2.reshape(desc2.shape[0], -1).T)

    rows = (s >= np.expand_dims(np.partition(s, -2)[:, -2], 1))
    cols = (s.T >= np.expand_dims(np.partition(s.T, -2)[:, -2], 1)).T
    threshold = (s > min_score)
    res = rows * cols * threshold
    return [np.where(res)[X], np.where(res)[Y]]


def apply_homography(pos1, H12):
    """
    Apply homography to inhomogenous points.
    :param pos1: An array with shape (N,2) of [x,y] point coordinates.
    :param H12: A 3x3 homography matrix.
    :return: An array with the same shape as pos1 with [x,y] point coordinates obtained from transforming pos1 using H12.
    """
    hom = to_homogenous(pos1)
    trans = np.dot(H12, hom.T).T

    new = np.zeros_like(pos1, dtype=np.float64)
    new += trans[:, [X, Y]]

    return new / trans[:, [Z, Z]]


def to_homogenous(pos):
    """
    Transfers a set of [x, y] points tp homogenous points i.e [x, y, 1]
    :param pos: An array with shape (N,2) of [x,y] point coordinates.
    :return: An array with shape (N,3) with [x,y,1] as the new coordinates
    """
    hom = np.zeros((pos.shape[ROWS], HOMOGENOUS_COORDS))
    hom[:, Z] += 1
    hom[:, X:Z] += pos
    return hom


def ransac_homography(points1, points2, num_iter, inlier_tol, translation_only=False):
    """
    Computes homography between two sets of points using RANSAC.
    :param points1: An array with shape (N,2) containing N rows of [x,y] coordinates of matched points in image 1.
    :param points2: An array with shape (N,2) containing N rows of [x,y] coordinates of matched points in image 2.
    :param num_iter: Number of RANSAC iterations to perform.
    :param inlier_tol: inlier tolerance threshold.
    :param translation_only: see estimate rigid transform
    :return: A list containing:
                1) A 3x3 normalized homography matrix.
                2) An Array with shape (S,) where S is the number of inliers,
                    containing the indices in pos1/pos2 of the maximal set of inlier matches found.
    """
    inliers = np.array([])

    for i in range(num_iter):
        if translation_only:
            indices = np.random.randint(points1.shape[ROWS], size=TRANS_ONLY)
        else:
            indices = np.random.randint(points1.shape[ROWS], size=TRANS_ROTATION)

        temp_mat = estimate_rigid_transform(points1[indices], points2[indices])
        temp_pts = apply_homography(points1, temp_mat)
        euclidean = np.linalg.norm(temp_pts - points2, axis=COLS) ** 2
        temp_inliers = np.where(euclidean < inlier_tol)[0]

        if temp_inliers.size > inliers.size:
            inliers = temp_inliers

    homography = estimate_rigid_transform(points1[inliers], points2[inliers], translation_only)
    return [homography, inliers]


def display_matches(im1, im2, points1, points2, inliers):
    """
    Display matching points.
    :param im1: A grayscale image.
    :param im2: A grayscale image.
    :param points1: An array shape (N,2), containing N rows of [x,y] coordinates of matched points in im1.
    :param points2: An array shape (N,2), containing N rows of [x,y] coordinates of matched points in im2.
    :param inliers: An array with shape (S,) of inlier matches.
    """
    image = np.hstack((im1, im2))

    translated_pts2 = points2
    translated_pts2[:, ROWS] += im1.shape[COLS]

    outliers1 = np.delete(points1, inliers, axis=ROWS)
    outliers2 = np.delete(translated_pts2, inliers, axis=ROWS)

    plt.figure()
    plt.imshow(image, cmap='gray')

    plt.plot((points1[inliers][:, ROWS], translated_pts2[inliers][:, ROWS]), (points1[inliers][:, COLS],
             translated_pts2[inliers][:, COLS]), mfc='r', c='y', lw=.8, ms=1, marker='o')

    plt.plot((outliers1[:, ROWS], outliers2[:, ROWS]), (outliers1[:, COLS], outliers2[:, COLS]),
             mfc='r', c='b', lw=.2, ms=1, marker='o')

    plt.show()


def accumulate_homographies(H_succesive, m):
    """
    Convert a list of succesive homographies to a list of homographies to a common reference frame.
    :param H_succesive: A list of M-1 3x3 homography matrices where H_successive[i] is a homography which transforms
           points from coordinate system i to coordinate system i+1.
    :param m: Index of the coordinate system towards which we would like to accumulate the given homographies.
    :return: A list of M 3x3 homography matrices, where H2m[i] transforms points from coordinate system i to
             coordinate system m
    """
    homographies = []

    for i in range(len(H_succesive) + 1):
        if i > m:
            homographies.append(get_acc_mat(H_succesive, m, i))

        else:
            homographies.append(get_acc_mat(H_succesive, m, i, smaller=True))

    return homographies


def get_acc_mat(H_successive, m, i, smaller=False):
    """
    Calculates a transformation accumulated to be from coordinate system i to coordinate system m
    :param H_successive: A list of M-1 3x3 homography matrices where H_successive[i] is a homography which transforms
           points from coordinate system i to coordinate system i+1.
    :param m: Index of the coordinate system towards which we would like to accumulate the given homographies.
    :param i: current coordinate system
    :param smaller: True if i <= m. default value is False
    :return: 3x3 homography matrix which transforms points from coordinate system i to coordinate system m
    """
    temp = np.eye(TRANS_MAT_SIZE)

    if smaller:
        for j in range(i, m):
            temp = H_successive[j].dot(temp)

    else:
        for j in range(i, m, -1):
            temp = np.linalg.inv(H_successive[j-1]).dot(temp)

    return temp / temp[2, 2]


def compute_bounding_box(homography, w, h):
    """
    computes bounding box of warped image under homography, without actually warping the image
    :param homography: homography
    :param w: width of the image
    :param h: height of the image
    :return: 2x2 array, where the first row is [x,y] of the top left corner,
     and the second row is the [x,y] of the bottom right corner
    """
    new_corners = apply_homography(np.array([[0, 0], [0, h], [w, 0], [w, h]]), homography)

    min_x = np.min(new_corners[:, X])
    max_x = np.max(new_corners[:, X])
    min_y = np.min(new_corners[:, Y])
    max_y = np.max(new_corners[:, Y])

    return np.array([[min_x, min_y], [max_x, max_y]], dtype=np.int64)


def warp_channel(image, homography):
    """
    Warps a 2D image with a given homography.
    :param image: a 2D image.
    :param homography: homograhpy.
    :return: A 2d warped image.
    """
    top_left, bottom_right = compute_bounding_box(homography, image.shape[COLS], image.shape[ROWS])

    x_vec = np.arange(top_left[X], bottom_right[X] + 1)
    y_vec = np.arange(top_left[Y], bottom_right[Y] + 1)

    x, y = np.meshgrid(x_vec, y_vec)
    pts = np.dstack((x, y))

    new_pts = apply_homography(pts.reshape(-1, 2), np.linalg.inv(homography))
    intensities = map_coordinates(image, [new_pts[:, Y], new_pts[:, X]], order=1, prefilter=False)
    return intensities.reshape(y_vec.size, x_vec.size)


def warp_image(image, homography):
    """
    Warps an RGB image with a given homography.
    :param image: an RGB image.
    :param homography: homograhpy.
    :return: A warped image.
    """
    return np.dstack([warp_channel(image[..., channel], homography) for channel in range(3)])


def filter_homographies_with_translation(homographies, minimum_right_translation):
    """
    Filters rigid transformations encoded as homographies by the amount of translation from left to right.
    :param homographies: homograhpies to filter.
    :param minimum_right_translation: amount of translation below which the transformation is discarded.
    :return: filtered homographies..
    """
    translation_over_thresh = [0]
    last = homographies[0][0, -1]
    for i in range(1, len(homographies)):
        if np.abs(homographies[i][0, -1] - last) > minimum_right_translation:
            translation_over_thresh.append(i)
            last = homographies[i][0, -1]
    return np.array(translation_over_thresh).astype(np.int)


def estimate_rigid_transform(points1, points2, translation_only=False):
    """
    Computes rigid transforming points1 towards points2, using least squares method.
    points1[i,:] corresponds to poins2[i,:]. In every point, the first coordinate is *x*.
    :param points1: array with shape (N,2). Holds coordinates of corresponding points from image 1.
    :param points2: array with shape (N,2). Holds coordinates of corresponding points from image 2.
    :param translation_only: whether to compute translation only. False (default) to compute rotation as well.
    :return: A 3x3 array with the computed homography.
    """
    centroid1 = points1.mean(axis=0)
    centroid2 = points2.mean(axis=0)

    if translation_only:
        rotation = np.eye(2)
        translation = centroid2 - centroid1

    else:
        centered_points1 = points1 - centroid1
        centered_points2 = points2 - centroid2

        sigma = centered_points2.T @ centered_points1
        U, _, Vt = np.linalg.svd(sigma)

        rotation = U @ Vt
        translation = -rotation @ centroid1 + centroid2

    H = np.eye(3)
    H[:2, :2] = rotation
    H[:2, 2] = translation
    return H


def non_maximum_suppression(image):
    """
    Finds local maximas of an image.
    :param image: A 2D array representing an image.
    :return: A boolean array with the same shape as the input image, where True indicates local maximum.
    """
    # Find local maximas.
    neighborhood = generate_binary_structure(2, 2)
    local_max = maximum_filter(image, footprint=neighborhood) == image
    local_max[image < (image.max() * 0.1)] = False

    # Erode areas to single points.
    lbs, num = label(local_max)
    centers = center_of_mass(local_max, lbs, np.arange(num) + 1)
    centers = np.stack(centers).round().astype(np.int)
    ret = np.zeros_like(image, dtype=np.bool)
    ret[centers[:, 0], centers[:, 1]] = True

    return ret


def spread_out_corners(im, m, n, radius):
    """
    Splits the image im to m by n rectangles and uses harris_corner_detector on each.
    :param im: A 2D array representing an image.
    :param m: Vertical number of rectangles.
    :param n: Horizontal number of rectangles.
    :param radius: Minimal distance of corner points from the boundary of the image.
    :return: An array with shape (N,2), where ret[i,:] are the [x,y] coordinates of the ith corner points.
    """
    corners = [np.empty((0, 2), dtype=np.int)]
    x_bound = np.linspace(0, im.shape[1], n + 1, dtype=np.int)
    y_bound = np.linspace(0, im.shape[0], m + 1, dtype=np.int)
    for i in range(n):
        for j in range(m):
            # Use Harris detector on every sub image.
            sub_im = im[y_bound[j]:y_bound[j + 1], x_bound[i]:x_bound[i + 1]]
            sub_corners = harris_corner_detector(sub_im)
            sub_corners += np.array([x_bound[i], y_bound[j]])[np.newaxis, :]
            corners.append(sub_corners)
    corners = np.vstack(corners)
    legit = ((corners[:, 0] > radius) & (corners[:, 0] < im.shape[1] - radius) &
             (corners[:, 1] > radius) & (corners[:, 1] < im.shape[0] - radius))
    ret = corners[legit, :]
    return ret


#################################################################################################################
################################################ SUPPLIED CODE ##################################################
#################################################################################################################

class PanoramicVideoGenerator:
    """
    Generates panorama from a set of images.
    """

    def __init__(self, data_dir, file_prefix, num_images):
        """
        The naming convention for a sequence of images is file_prefixN.jpg,
        where N is a running number 001, 002, 003...
        :param data_dir: path to input images.
        :param file_prefix: see above.
        :param num_images: number of images to produce the panoramas with.
        """
        self.file_prefix = file_prefix
        self.files = [os.path.join(data_dir, '%s%03d.jpg' % (file_prefix, i + 1)) for i in range(num_images)]
        self.files = list(filter(os.path.exists, self.files))
        self.panoramas = None
        self.homographies = None
        self.images = []
        print('found %d images' % len(self.files))

    def align_images(self, translation_only=False):
        """
        compute homographies between all images to a common coordinate system
        :param translation_only: see estimte_rigid_transform
        """
        # Extract feature point locations and descriptors.
        points_and_descriptors = []
        for file in self.files:
            image = sol4_utils.read_image(file, 1)
            self.images.append(image)
            self.h, self.w = image.shape
            pyramid, _ = sol4_utils.build_gaussian_pyramid(image, 3, 7)
            points_and_descriptors.append(find_features(pyramid))

        # Compute homographies between successive pairs of images.
        Hs = []
        for i in range(len(points_and_descriptors) - 1):
            points1, points2 = points_and_descriptors[i][0], points_and_descriptors[i + 1][0]
            desc1, desc2 = points_and_descriptors[i][1], points_and_descriptors[i + 1][1]

            # Find matching feature points.
            ind1, ind2 = match_features(desc1, desc2, .7)
            points1, points2 = points1[ind1, :], points2[ind2, :]

            # Compute homography using RANSAC.
            H12, inliers = ransac_homography(points1, points2, NUM_OF_ITER, INLIER_TOL, translation_only)

            # Uncomment for debugging: display inliers and outliers among matching points.
            # In the submitted code this function should be commented out!
            # display_matches(self.images[i], self.images[i+1], points1 , points2, inliers)
            Hs.append(H12)
        # Compute composite homographies from the central coordinate system.
        accumulated_homographies = accumulate_homographies(Hs, (len(Hs) - 1) // 2)
        self.homographies = np.stack(accumulated_homographies)
        self.frames_for_panoramas = filter_homographies_with_translation(self.homographies, minimum_right_translation=5)
        self.homographies = self.homographies[self.frames_for_panoramas]

    def generate_panoramic_images(self, number_of_panoramas):
        """
        combine slices from input images to panoramas.
        :param number_of_panoramas: how many different slices to take from each input image
        """
        assert self.homographies is not None

        # compute bounding boxes of all warped input images in the coordinate system of the middle image (as given by the homographies)
        self.bounding_boxes = np.zeros((self.frames_for_panoramas.size, 2, 2))
        for i in range(self.frames_for_panoramas.size):
            self.bounding_boxes[i] = compute_bounding_box(self.homographies[i], self.w, self.h)

        # change our reference coordinate system to the panoramas
        # all panoramas share the same coordinate system
        global_offset = np.min(self.bounding_boxes, axis=(0, 1))
        self.bounding_boxes -= global_offset

        slice_centers = np.linspace(0, self.w, number_of_panoramas + 2, endpoint=True, dtype=np.int)[1:-1]
        warped_slice_centers = np.zeros((number_of_panoramas, self.frames_for_panoramas.size))
        # every slice is a different panorama, it indicates the slices of the input images from which the panorama
        # will be concatenated
        for i in range(slice_centers.size):
            slice_center_2d = np.array([slice_centers[i], self.h // 2])[None, :]
            # homography warps the slice center to the coordinate system of the middle image
            warped_centers = [apply_homography(slice_center_2d, h) for h in self.homographies]
            # we are actually only interested in the x coordinate of each slice center in the panoramas' coordinate system
            warped_slice_centers[i] = np.array(warped_centers)[:, :, 0].squeeze() - global_offset[0]

        panorama_size = np.max(self.bounding_boxes, axis=(0, 1)).astype(np.int) + 1

        # boundary between input images in the panorama
        x_strip_boundary = ((warped_slice_centers[:, :-1] + warped_slice_centers[:, 1:]) / 2)
        x_strip_boundary = np.hstack([np.zeros((number_of_panoramas, 1)),
                                      x_strip_boundary,
                                      np.ones((number_of_panoramas, 1)) * panorama_size[0]])
        x_strip_boundary = x_strip_boundary.round().astype(np.int)

        self.panoramas = np.zeros((number_of_panoramas, panorama_size[1], panorama_size[0], 3), dtype=np.float64)
        for i, frame_index in enumerate(self.frames_for_panoramas):
            # warp every input image once, and populate all panoramas
            image = sol4_utils.read_image(self.files[frame_index], 2)
            warped_image = warp_image(image, self.homographies[i])
            x_offset, y_offset = self.bounding_boxes[i][0].astype(np.int)
            y_bottom = y_offset + warped_image.shape[0]

            for panorama_index in range(number_of_panoramas):
                # take strip of warped image and paste to current panorama
                boundaries = x_strip_boundary[panorama_index, i:i + 2]
                image_strip = warped_image[:, boundaries[0] - x_offset: boundaries[1] - x_offset]
                x_end = boundaries[0] + image_strip.shape[1]
                self.panoramas[panorama_index, y_offset:y_bottom, boundaries[0]:x_end] = image_strip

        # crop out areas not recorded from enough angles
        # assert will fail if there is overlap in field of view between the left most image and the right most image
        crop_left = int(self.bounding_boxes[0][1, 0])
        crop_right = int(self.bounding_boxes[-1][0, 0])
        if crop_left < crop_right:  # 'for testing your code with a few images do not crop.'
            self.panoramas = self.panoramas[:, :, crop_left:crop_right, :]

    def save_panoramas_to_video(self):
        assert self.panoramas is not None
        out_folder = 'tmp_folder_for_panoramic_frames/%s' % self.file_prefix
        try:
            shutil.rmtree(out_folder)
        except:
            print('could not remove folder')
            pass
        os.makedirs(out_folder)
        # save individual panorama images to 'tmp_folder_for_panoramic_frames'
        for i, panorama in enumerate(self.panoramas):
            imwrite('%s/panorama%02d.png' % (out_folder, i + 1), panorama)
        if os.path.exists('%s.mp4' % self.file_prefix):
            os.remove('%s.mp4' % self.file_prefix)
        # write output video to current folder
        os.system('ffmpeg -framerate 15 -i %s/panorama%%02d.png %s.mp4' %
                  (out_folder, self.file_prefix))

    def show_panorama(self, panorama_index, figsize=(20, 20)):
        assert self.panoramas is not None
        plt.figure(figsize=figsize)
        plt.imshow(self.panoramas[panorama_index].clip(0, 1))
        plt.show()
