import io
import os
import numpy as np
import cv2
from scipy.ndimage import morphology
from skimage import measure
from skimage.morphology import thin
from skimage.measure import regionprops
from scipy.stats import kurtosis
from scipy.stats import skew
from skimage.morphology import convex_hull_image
import pandas as pd



def load_and_skeletonize_image(file_path):
    image = io.imread(file_path)
    binary_image = image > 0
    skeleton = morphology.skeletonize(binary_image)
    return skeleton

def eight_neighbors(x, y, image):
        VIII_neighbors = [image[x, y - 1], image[x - 1, y - 1], image[x - 1, y], image[x - 1, y + 1],
                          image[x, y + 1], image[x + 1, y + 1], image[x + 1, y], image[x + 1, y - 1]]
        return VIII_neighbors

def getSkeletonIntersection(skeleton):

    validIntersection = [[0, 1, 0, 1, 0, 0, 1, 0], [0, 0, 1, 0, 1, 0, 0, 1], [1, 0, 0, 1, 0, 1, 0, 0],
                         [0, 1, 0, 0, 1, 0, 1, 0], [0, 0, 1, 0, 0, 1, 0, 1], [1, 0, 0, 1, 0, 0, 1, 0],
                         [0, 1, 0, 0, 1, 0, 0, 1], [1, 0, 1, 0, 0, 1, 0, 0], [0, 1, 0, 0, 0, 1, 0, 1],
                         [0, 1, 0, 1, 0, 0, 0, 1], [0, 1, 0, 1, 0, 1, 0, 0], [0, 0, 0, 1, 0, 1, 0, 1],
                         [1, 0, 1, 0, 0, 0, 1, 0], [1, 0, 1, 0, 1, 0, 0, 0], [0, 0, 1, 0, 1, 0, 1, 0],
                         [1, 0, 0, 0, 1, 0, 1, 0], [1, 0, 0, 1, 1, 1, 0, 0], [0, 0, 1, 0, 0, 1, 1, 1],
                         [1, 1, 0, 0, 1, 0, 0, 1], [0, 1, 1, 1, 0, 0, 1, 0], [1, 0, 1, 1, 0, 0, 1, 0],
                         [1, 0, 1, 0, 0, 1, 1, 0], [1, 0, 1, 1, 0, 1, 1, 0], [0, 1, 1, 0, 1, 0, 1, 1],
                         [1, 1, 0, 1, 1, 0, 1, 0], [1, 1, 0, 0, 1, 0, 1, 0], [0, 1, 1, 0, 1, 0, 1, 0],
                         [0, 0, 1, 0, 1, 0, 1, 1], [1, 0, 0, 1, 1, 0, 1, 0], [1, 0, 1, 0, 1, 1, 0, 1],
                         [1, 0, 1, 0, 1, 1, 0, 0], [1, 0, 1, 0, 1, 0, 0, 1], [0, 1, 0, 0, 1, 0, 1, 1],
                         [0, 1, 1, 0, 1, 0, 0, 1], [1, 1, 0, 1, 0, 0, 1, 0], [0, 1, 0, 1, 1, 0, 1, 0],
                         [0, 0, 1, 0, 1, 1, 0, 1], [1, 0, 1, 0, 0, 1, 0, 1], [1, 0, 0, 1, 0, 1, 1, 0],
                         [1, 0, 1, 1, 0, 1, 0, 0]];
    image = skeleton.copy();
    image = image / 255;
    intersections = list();
    for x in range(1, len(image) - 1):
        for y in range(1, len(image[x]) - 1):
            if image[x][y] == 1:
                neighbours = eight_neighbors(x, y, image);
                if neighbours in validIntersection:
                    intersections.append((y, x));

    for point1 in intersections:
        for point2 in intersections:
            if (((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2) < 10 ** 2) and (point1 != point2):
                intersections.remove(point2);

    intersections = list(set(intersections));
    return intersections;

def measurement(directory_path, save_path):
    database = pd.DataFrame([[0] * 103],
                            columns=['cell_name', 'cell_mean_mito_area_(pixels_squared)', 'cell_median_mito_area_(pixels_squared)',
                                     'cell_std_mito_area_(pixels_squared)', 'cell_mean_mito_eccentricity',
                                     'cell_median_mito_eccentricity', 'cell_std_mito_eccentricity',
                                     'cell_mean_mito_equi_diameter_(pixels)', 'cell_median_mito_equi_diameter_(pixels)',
                                     'cell_std_mito_equi_diameter_(pixels)', 'cell_mean_mito_euler_number','cell_std_mito_euler_number',
                                     'cell_mean_mito_extent',
                                     'cell_median_mito_extent', 'cell_std_mito_extent',
                                     'cell_mean_mito_major_axis_(pixels)',
                                     'cell_median_mito_major_axis_(pixels)', 'cell_std_mito_major_axis_(pixels)',
                                     'cell_mean_mito_minor_axis_(pixels)',
                                     'cell_median_mito_minor_axis_(pixels)', 'cell_std_mito_minor_axis_(pixels)',
                                     'cell_mean_mito_orientation_(degrees)',
                                     'cell_median_mito_orientation_(degrees)', 'cell_std_mito_orientation_(degrees)',
                                     'cell_mean_mito_perimeter_(pixels)',
                                     'cell_median_mito_perimeter_(pixels)', 'cell_std_mito_perimeter_(pixels)',
                                     'cell_mean_mito_solidity',
                                     'cell_median_mito_solidity', 'cell_std_mito_solidity',
                                     'cell_mean_mito_centroid_x_(pixels)',
                                     'cell_median_mito_centroid_x_(pixels)', 'cell_std_mito_centroid_x_(pixels)',
                                     'cell_mean_mito_centroid_y_(pixels)',
                                     'cell_median_mito_centroid_y_(pixels)', 'cell_std_mito_centroid_y_(pixels)',
                                     'cell_mean_mito_distance_(pixels)',
                                     'cell_median_mito_distance_(pixels)', 'cell_std_mito_distance_(pixels)',
                                     'cell_mean_mito_weighted_cent_x_(pixels)',
                                     'cell_median_mito_weighted_cent_x_(pixels)',
                                     'cell_std_mito_weighted_cent_x_(pixels)',
                                     'cell_mean_mito_weighted_cent_y_(pixels)',
                                     'cell_median_mito_weighted_cent_y_(pixels)',
                                     'cell_std_mito_weighted_cent_y_(pixels)',
                                     'cell_mean_mito_weighted_distance_(pixels)',
                                     'cell_median_mito_weighted_distance_(pixels)',
                                     'cell_std_mito_weighted_distance_(pixels)',
                                     'cell_mean_mito_form_factor', 'cell_median_mito_form_factor',
                                     'cell_std_mito_form_factor', 'cell_mean_mito_roundness',
                                     'cell_median_mito_roundness',
                                     'cell_std_mito_roundness', 'cell_mean_mito_branch_count',
                                     'cell_std_mito_branch_count', 'cell_mean_mito_mean_branch_length_(pixels)',
                                     'cell_median_mito_mean_branch_length_(pixels)',
                                     'cell_std_mito_mean_branch_length_(pixels)',
                                     'cell_mean_mito_total_branch_length_(pixels)',
                                     'cell_median_mito_total_branch_length_(pixels)',
                                     'cell_std_mito_total_branch_length_(pixels)',
                                     'cell_mean_mito_median_branch_length_(pixels)',
                                     'cell_median_mito_median_branch_length_(pixels)',
                                     'cell_std_mito_median_branch_length_(pixels)',
                                     'cell_mean_mito_std_branch_length_(pixels)',
                                     'cell_std_mito_std_branch_length_(degrees)',
                                     'cell_mean_mito_mean_branch_angle_(degrees)',
                                     'cell_median_mito_mean_branch_angle_(degrees)',
                                     'cell_std_mito_mean_branch_angle_(degrees)',
                                     'cell_mean_mito_median_branch_angle_(degrees)',
                                     'cell_median_mito_median_branch_angle_(degrees)',
                                     'cell_std_mito_median_branch_angle_(degrees)',
                                     'cell_mean_mito_std_branch_angle_(degrees)',
                                     'cell_std_mito_std_branch_angle_(degrees)',
                                     'cell_mean_mito_total_density_(pixels)', 'cell_median_mito_total_density_(pixels)',
                                     'cell_std_mito_total_density', 'cell_mean_mito_average_density_(pixels)',
                                     'cell_median_mito_average_density_(pixels)',
                                     'cell_std_mito_average_density_(pixels)',
                                     'cell_mean_mito_median_density_(pixels)', 'cell_median_mito_median_density',
                                     'cell_std_mito_median_density_(pixels)', 'cell_kurtosis_x',
                                     'cell_weighted_kurtosis_x',
                                     'cell_kurtosis_y', 'cell_weighted_kurtosis_y', 'cell_kurtosis_squared',
                                     'cell_weighted_kurtosis_squared', 'cell_skewness_x', 'cell_weighted_skewness_x',
                                     'cell_skewness_y', 'cell_weighted_skewness_y', 'cell_skewness_squared',
                                     'cell_weighted_skewness_squared', 'cell_network_orientation_(degrees)',
                                     'cell_network_major_axis_(pixels)',
                                     'cell_network_minor_axis_(pixels)', 'cell_network_eccentricity',
                                     'cell_network_effective_extent', 'cell_network_effective_solidity',
                                     'cell_network_fractal_dimension'])

    database_raw = pd.DataFrame([[0] * 39], columns=['cell_name', 'resize_factor', 'mito_area', 'mito_centroid',
                                                     'mito_eccentricity',
                                                     'mito_equi_diameter', 'mito_euler_number', 'mito_extent',
                                                     'mito_major_axis',
                                                     'mito_minor_axis', 'mito_orientation', 'mito_perimeter',
                                                     'mito_solidity',
                                                     'mito_centroid_x', 'mito_centroid_y', 'mito_distance',
                                                     'mito_weighted_cent_x',
                                                     'mito_weighted_cent_y', 'mito_weighted_distance',
                                                     'mito_form_factor',
                                                     'mito_roundness', 'mito_branch_count', 'mito_total_branch_length',
                                                     'mito_mean_branch_length', 'mito_median_branch_length',
                                                     'mito_std_branch_length',
                                                     'mito_mean_branch_angle', 'mito_median_branch_angle',
                                                     'mito_std_branch_angle',
                                                     'mito_total_density', 'mito_average_density',
                                                     'mito_median_density',
                                                     'mito_branch_count', 'mito_distance', 'mito_weighted_cent_x',
                                                     'mito_weighted_cent_y',
                                                     'mito_weighted_distance', 'mito_form_factor', 'mito_roundness'])

    test_num = 0
    files = os.listdir(directory_path)  # 获取目录中的文件列表
    total_file_count = len(files)
    image_extensions = ['.jpg', '.png', '.tif', '.tiff']

    for file in files:
        test_num += 1
        if any(file.lower().endswith(ext) for ext in image_extensions):
            try:
                file_path = os.path.join(directory_path, file)
                img = cv2.imread(file_path)
                img = img[:,:,0]
                print("Test", file, f'Test [{np.round(100*(test_num/total_file_count),2)}%]')
                scale =1

                #独立线粒体测试
                mito_labels = measure.label(np.array(img), connectivity=2)
                mito_props = regionprops(mito_labels)

                mito_area = []
                mito_centroid = []
                mito_eccentricity = []
                mito_equi_diameter = []
                mito_euler_number = []
                mito_extent = []
                mito_major_axis = []
                mito_minor_axis = []
                mito_orientation = []
                mito_perimeter = []
                mito_solidity = []
                mito_centroid_x = []
                mito_centroid_y = []
                mito_distance = []
                mito_weighted_cent_x = []
                mito_weighted_cent_y = []
                mito_weighted_distance = []
                mito_form_factor = []
                mito_roundness = []
                mito_branch_count = []
                mito_total_branch_length = []
                mito_mean_branch_length = []
                mito_median_branch_length = []
                mito_std_branch_length = []
                mito_mean_branch_angle = []
                mito_median_branch_angle = []
                mito_std_branch_angle = []
                mito_total_density = []
                mito_average_density = []
                mito_median_density = []
                mito_branch_count = []

                for r in range(len(mito_props)):
                    if mito_props[r].area > 16:

                        mito_area.append(mito_props[r].area)
                        mito_eccentricity.append(mito_props[r].eccentricity)
                        mito_equi_diameter.append(mito_props[r].equivalent_diameter)
                        mito_euler_number.append(mito_props[r].euler_number)
                        mito_extent.append(mito_props[r].extent)
                        mito_major_axis.append(mito_props[r].major_axis_length)
                        mito_minor_axis.append(mito_props[r].minor_axis_length)
                        mito_orientation.append(mito_props[r].orientation)
                        mito_perimeter.append(mito_props[r].perimeter)
                        mito_solidity.append(mito_props[r].solidity)
                        mito_centroid.append(mito_props[r].centroid)
                        mito_centroid_x.append(mito_props[r].centroid[0])
                        mito_centroid_y.append(mito_props[r].centroid[1])

                        if mito_props[r].label == 0:
                            continue

                        else:
                            labelMask = np.zeros(img.shape, dtype="uint8")
                            labelMask[mito_labels == mito_props[r].label] = 255

                            BranchPointsPositions = []
                            branch_points_ctr = []
                            num_branch_points = 0
                            number_branches = 0
                            branch_length = []
                            branch_angle = []

                            try:
                                imagebw8 = labelMask
                                imagebw8 = imagebw8.astype(np.int32)
                                Skel2 = thin(labelMask)
                                Skel2 = 255 * Skel2
                                branch_pointsn = getSkeletonIntersection(Skel2)
                                number_branchpoints = len(branch_pointsn)

                                bp = np.zeros(shape=(imagebw8.shape[0], imagebw8.shape[1]), dtype=np.uint8)
                                for ii in range(len(branch_pointsn)):
                                    xi = branch_pointsn[ii][0]
                                    yi = branch_pointsn[ii][1]
                                    BranchPointsPositions.append([yi, xi])
                                    bp[yi, xi] = 255

                                kernelbp = np.ones((3, 3), np.uint8)
                                IM = cv2.dilate(bp, kernelbp, iterations=1)

                                BranchLengthMatrix = Skel2 - IM
                                BranchMatrix = BranchLengthMatrix > 0

                                imagebwlabels2 = measure.label(np.array(BranchMatrix), connectivity=2)
                                NUMimagebw1 = imagebwlabels2.max()

                                propsbmm = regionprops(imagebwlabels2)

                                dist_transform2 = cv2.distanceTransform(labelMask, cv2.DIST_L2, 5)
                                for rq in range(len(propsbmm)):
                                    branch_length.append((propsbmm[rq].area) + 4)
                                    branch_angle.append(propsbmm[rq].orientation)

                                if len(branch_length) == 1:
                                    branch_length[0] = mito_props[r].major_axis_length

                                branch_angle = np.multiply(branch_angle, (180 / np.pi))
                                num_branch_points = number_branchpoints
                                number_branches = NUMimagebw1
                                total_branch_length = np.sum(branch_length)
                                mean_branch_length = np.mean(branch_length)
                                median_branch_length = np.median(branch_length)
                                std_branch_length = np.std(branch_length)
                                mean_branch_angle = np.mean(branch_angle)
                                median_branch_angle = np.median(branch_angle)
                                std_branch_angle = np.std(branch_angle)

                            except:
                                branch_points_ctr.append([])
                                num_branch_points = 0
                                number_branches = 0
                                branch_length.append([])
                                branch_angle.append([])

                                total_branch_length = 0
                                mean_branch_length = 0
                                median_branch_length = 0
                                std_branch_length = 0
                                mean_branch_angle = 0
                                median_branch_angle = 0
                                std_branch_angle = 0

                    if mito_props[r].area > 16:
                        mito_branch_count.append(number_branches)

                        mito_total_branch_length.append(total_branch_length)
                        mito_mean_branch_length.append(mean_branch_length)
                        mito_median_branch_length.append(median_branch_length)
                        mito_std_branch_length.append(std_branch_length)
                        mito_mean_branch_angle.append(mean_branch_angle)
                        mito_median_branch_angle.append(median_branch_angle)
                        mito_std_branch_angle.append(std_branch_angle)

                        mito_total_density.append(np.sum(dist_transform2[dist_transform2 > 0]))
                        mito_average_density.append(np.mean(dist_transform2[dist_transform2 > 0]))
                        mito_median_density.append(np.median(dist_transform2[dist_transform2 > 0]))

                mito_area = np.multiply(np.power(scale, 2), mito_area)
                mito_equi_diameter = np.multiply(scale, mito_equi_diameter)
                mito_major_axis = np.multiply(scale, mito_major_axis)
                mito_minor_axis = np.multiply(scale, mito_minor_axis)
                mito_perimeter = np.multiply(scale, mito_perimeter)
                mito_centroid_x = np.multiply(scale, mito_centroid_x)
                mito_centroid_y = np.multiply(scale, mito_centroid_y)

                mito_distance = np.sqrt(np.power(mito_centroid_x, 2) + np.power(mito_centroid_y, 2))
                mito_weighted_cent_x = np.divide(np.multiply(mito_centroid_x, mito_area), np.sum(mito_area))
                mito_weighted_cent_y = np.divide(np.multiply(mito_centroid_y, mito_area), np.sum(mito_area))
                mito_weighted_distance = np.sqrt(np.power(mito_weighted_cent_x, 2) + np.power(mito_weighted_cent_y, 2))
                mito_form_factor = (np.divide(np.power(mito_perimeter, 2), mito_area)) / (4 * np.pi)
                mito_roundness = ((4 / np.pi) * np.divide(mito_area, np.power(mito_major_axis, 2)))

                mito_total_branch_length = np.multiply(scale, mito_total_branch_length)
                mito_mean_branch_length = np.multiply(scale, mito_mean_branch_length)
                mito_median_branch_length = np.multiply(scale, mito_median_branch_length)
                mito_std_branch_length = np.multiply(scale, mito_std_branch_length)
                mito_total_density = np.multiply(scale, mito_total_density)
                mito_average_density = np.multiply(scale, mito_average_density)
                mito_median_density = np.multiply(scale, mito_median_density)

                # 单张图片为单位
                cell_mito_count = len(mito_area)
                cell_total_mito_area = np.sum(mito_area)
                cell_mean_mito_area = np.mean(mito_area)
                cell_median_mito_area = np.median(mito_area)
                cell_std_mito_area = np.std(mito_area)
                cell_mean_mito_eccentricity = np.mean(mito_eccentricity)
                cell_median_mito_eccentricity = np.median(mito_eccentricity)
                cell_std_mito_eccentricity = np.std(mito_eccentricity)
                cell_mean_mito_equi_diameter = np.mean(mito_equi_diameter)
                cell_median_mito_equi_diameter = np.median(mito_equi_diameter)
                cell_std_mito_equi_diameter = np.std(mito_equi_diameter)
                cell_mean_mito_euler_number = np.mean(mito_euler_number)
                cell_median_mito_euler_number = np.median(mito_euler_number)
                cell_std_mito_euler_number = np.std(mito_euler_number)
                cell_mean_mito_extent = np.mean(mito_extent)
                cell_median_mito_extent = np.median(mito_extent)
                cell_std_mito_extent = np.std(mito_extent)
                cell_mean_mito_major_axis = np.mean(mito_major_axis)
                cell_median_mito_major_axis = np.median(mito_major_axis)
                cell_std_mito_major_axis = np.std(mito_major_axis)
                cell_mean_mito_minor_axis = np.mean(mito_minor_axis)
                cell_median_mito_minor_axis = np.median(mito_minor_axis)
                cell_std_mito_minor_axis = np.std(mito_minor_axis)
                cell_mean_mito_orientation = np.mean(mito_orientation)
                cell_median_mito_orientation = np.median(mito_orientation)
                cell_std_mito_orientation = np.std(mito_orientation)
                cell_mean_mito_perimeter = np.mean(mito_perimeter)
                cell_median_mito_perimeter = np.median(mito_perimeter)
                cell_std_mito_perimeter = np.std(mito_perimeter)
                cell_mean_mito_solidity = np.mean(mito_solidity)
                cell_median_mito_solidity = np.median(mito_solidity)
                cell_std_mito_solidity = np.std(mito_solidity)
                cell_mean_mito_centroid_x = np.mean(mito_centroid_x)
                cell_median_mito_centroid_x = np.median(mito_centroid_x)
                cell_std_mito_centroid_x = np.std(mito_centroid_x)
                cell_mean_mito_centroid_y = np.mean(mito_centroid_y)
                cell_median_mito_centroid_y = np.median(mito_centroid_y)
                cell_std_mito_centroid_y = np.std(mito_centroid_y)
                cell_mean_mito_distance = np.mean(mito_distance)
                cell_median_mito_distance = np.median(mito_distance)
                cell_std_mito_distance = np.std(mito_distance)
                cell_mean_mito_weighted_cent_x = np.mean(mito_weighted_cent_x)
                cell_median_mito_weighted_cent_x = np.median(mito_weighted_cent_x)
                cell_std_mito_weighted_cent_x = np.std(mito_weighted_cent_x)
                cell_mean_mito_weighted_cent_y = np.mean(mito_weighted_cent_y)
                cell_median_mito_weighted_cent_y = np.median(mito_weighted_cent_y)
                cell_std_mito_weighted_cent_y = np.std(mito_weighted_cent_y)
                cell_mean_mito_weighted_distance = np.mean(mito_weighted_distance)
                cell_median_mito_weighted_distance = np.median(mito_weighted_distance)
                cell_std_mito_weighted_distance = np.std(mito_weighted_distance)
                cell_mean_mito_form_factor = np.mean(mito_form_factor)
                cell_median_mito_form_factor = np.median(mito_form_factor)
                cell_std_mito_form_factor = np.std(mito_form_factor)
                cell_mean_mito_roundness = np.mean(mito_roundness)
                cell_median_mito_roundness = np.median(mito_roundness)
                cell_std_mito_roundness = np.std(mito_roundness)
                cell_mean_mito_branch_count = np.mean(mito_branch_count)
                cell_median_mito_branch_count = np.median(mito_branch_count)
                cell_std_mito_branch_count = np.std(mito_branch_count)
                cell_mean_mito_mean_branch_length = np.mean(mito_mean_branch_length)
                cell_median_mito_mean_branch_length = np.median(mito_mean_branch_length)
                cell_std_mito_mean_branch_length = np.std(mito_mean_branch_length)
                cell_mean_mito_total_branch_length = np.mean(mito_total_branch_length)
                cell_median_mito_total_branch_length = np.median(mito_total_branch_length)
                cell_std_mito_total_branch_length = np.std(mito_total_branch_length)
                cell_mean_mito_median_branch_length = np.mean(mito_median_branch_length)
                cell_median_mito_median_branch_length = np.median(mito_median_branch_length)
                cell_std_mito_median_branch_length = np.std(mito_median_branch_length)
                cell_mean_mito_std_branch_length = np.mean(mito_std_branch_length)
                cell_median_mito_std_branch_length = np.median(mito_std_branch_length)
                cell_std_mito_std_branch_length = np.std(mito_std_branch_length)
                cell_mean_mito_mean_branch_angle = np.mean(mito_mean_branch_angle)
                cell_median_mito_mean_branch_angle = np.median(mito_mean_branch_angle)
                cell_std_mito_mean_branch_angle = np.std(mito_mean_branch_angle)
                cell_mean_mito_median_branch_angle = np.mean(mito_median_branch_angle)
                cell_median_mito_median_branch_angle = np.median(mito_median_branch_angle)
                cell_std_mito_median_branch_angle = np.std(mito_median_branch_angle)
                cell_mean_mito_std_branch_angle = np.mean(mito_std_branch_angle)
                cell_median_mito_std_branch_angle = np.median(mito_std_branch_angle)
                cell_std_mito_std_branch_angle = np.std(mito_std_branch_angle)
                cell_mean_mito_total_density = np.mean(mito_total_density)
                cell_median_mito_total_density = np.median(mito_total_density)
                cell_std_mito_total_density = np.std(mito_total_density)
                cell_mean_mito_average_density = np.mean(mito_average_density)
                cell_median_mito_average_density = np.median(mito_average_density)
                cell_std_mito_average_density = np.std(mito_average_density)
                cell_mean_mito_median_density = np.mean(mito_median_density)
                cell_median_mito_median_density = np.median(mito_median_density)
                cell_std_mito_median_density = np.std(mito_median_density)
                cell_kurtosis_x = kurtosis(mito_centroid_x)
                cell_weighted_kurtosis_x = kurtosis(mito_weighted_cent_x)
                cell_kurtosis_y = kurtosis(mito_centroid_y)
                cell_weighted_kurtosis_y = kurtosis(mito_weighted_cent_y)
                cell_kurtosis_squared = np.add(np.power(cell_kurtosis_x, 2), np.power(cell_kurtosis_y, 2))
                cell_weighted_kurtosis_squared = np.add(np.power(cell_weighted_kurtosis_x, 2),
                                                        np.power(cell_weighted_kurtosis_y, 2))
                cell_skewness_x = skew(mito_centroid_x)
                cell_weighted_skewness_x = skew(mito_weighted_cent_x)
                cell_skewness_y = skew(mito_centroid_y)
                cell_weighted_skewness_y = skew(mito_weighted_cent_y)
                cell_skewness_squared = np.add(np.power(cell_skewness_x, 2), np.power(cell_skewness_y, 2))
                cell_weighted_skewness_squared = np.add(np.power(cell_weighted_skewness_x, 2),
                                                        np.power(cell_weighted_skewness_y, 2))

                chull = convex_hull_image(img)

                cell_labels = measure.label(np.array(chull), connectivity=2)
                cell_props = regionprops(cell_labels)
                cell_network_orientation = cell_props[0].orientation * 180 / np.pi
                cell_network_major_axis = cell_props[0].major_axis_length
                cell_network_minor_axis = cell_props[0].minor_axis_length
                cell_network_eccentricity = cell_props[0].eccentricity

                cell_scaled_area = np.multiply(np.power(scale, 2), cell_props[0].area)
                cell_network_effective_extent = (np.sum(mito_area) / cell_scaled_area) * cell_props[0].extent
                cell_network_effective_solidity = np.sum(mito_area) / cell_scaled_area

                cell_network_major_axis = np.multiply(scale, cell_network_major_axis)
                cell_network_minor_axis = np.multiply(scale, cell_network_minor_axis)

                pixels = []

                for i in range(img.shape[0]):
                    for j in range(img.shape[1]):
                        if img[i, j] > 0:
                            pixels.append((i, j))

                Lx = img.shape[1]
                Ly = img.shape[0]

                pixels = np.array(pixels)

                scales = np.logspace(0.01, 1, num=10, endpoint=False, base=2)
                Ns = []
                for scale1 in scales:
                    H, edges = np.histogramdd(pixels, bins=(np.arange(0, Lx, scale1), np.arange(0, Ly, scale1)))
                    Ns.append(np.sum(H > 0))

                coeffs = np.polyfit(np.log(scales), np.log(Ns), 1)
                cell_network_fractal_dimension = -coeffs[0]

                temp_dataset = pd.DataFrame([[file, cell_mean_mito_area,
                                              cell_median_mito_area, cell_std_mito_area, cell_mean_mito_eccentricity,
                                              cell_median_mito_eccentricity, cell_std_mito_eccentricity,
                                              cell_mean_mito_equi_diameter, cell_median_mito_equi_diameter,
                                              cell_std_mito_equi_diameter, cell_mean_mito_euler_number, cell_std_mito_euler_number,
                                              cell_mean_mito_extent, cell_median_mito_extent, cell_std_mito_extent,
                                              cell_mean_mito_major_axis, cell_median_mito_major_axis,
                                              cell_std_mito_major_axis,
                                              cell_mean_mito_minor_axis, cell_median_mito_minor_axis,
                                              cell_std_mito_minor_axis,
                                              cell_mean_mito_orientation, cell_median_mito_orientation,
                                              cell_std_mito_orientation,
                                              cell_mean_mito_perimeter, cell_median_mito_perimeter,
                                              cell_std_mito_perimeter,
                                              cell_mean_mito_solidity, cell_median_mito_solidity,
                                              cell_std_mito_solidity,
                                              cell_mean_mito_centroid_x, cell_median_mito_centroid_x,
                                              cell_std_mito_centroid_x,
                                              cell_mean_mito_centroid_y, cell_median_mito_centroid_y,
                                              cell_std_mito_centroid_y,
                                              cell_mean_mito_distance, cell_median_mito_distance,
                                              cell_std_mito_distance,
                                              cell_mean_mito_weighted_cent_x, cell_median_mito_weighted_cent_x,
                                              cell_std_mito_weighted_cent_x, cell_mean_mito_weighted_cent_y,
                                              cell_median_mito_weighted_cent_y, cell_std_mito_weighted_cent_y,
                                              cell_mean_mito_weighted_distance, cell_median_mito_weighted_distance,
                                              cell_std_mito_weighted_distance, cell_mean_mito_form_factor,
                                              cell_median_mito_form_factor, cell_std_mito_form_factor,
                                              cell_mean_mito_roundness, cell_median_mito_roundness,
                                              cell_std_mito_roundness,
                                              cell_mean_mito_branch_count,
                                              cell_std_mito_branch_count, cell_mean_mito_mean_branch_length,
                                              cell_median_mito_mean_branch_length, cell_std_mito_mean_branch_length,
                                              cell_mean_mito_total_branch_length, cell_median_mito_total_branch_length,
                                              cell_std_mito_total_branch_length, cell_mean_mito_median_branch_length,
                                              cell_median_mito_median_branch_length, cell_std_mito_median_branch_length,
                                              cell_mean_mito_std_branch_length,
                                              cell_std_mito_std_branch_length, cell_mean_mito_mean_branch_angle,
                                              cell_median_mito_mean_branch_angle, cell_std_mito_mean_branch_angle,
                                              cell_mean_mito_median_branch_angle, cell_median_mito_median_branch_angle,
                                              cell_std_mito_median_branch_angle, cell_mean_mito_std_branch_angle,
                                              cell_std_mito_std_branch_angle,
                                              cell_mean_mito_total_density, cell_median_mito_total_density,
                                              cell_std_mito_total_density, cell_mean_mito_average_density,
                                              cell_median_mito_average_density, cell_std_mito_average_density,
                                              cell_mean_mito_median_density, cell_median_mito_median_density,
                                              cell_std_mito_median_density, cell_kurtosis_x, cell_weighted_kurtosis_x,
                                              cell_kurtosis_y, cell_weighted_kurtosis_y, cell_kurtosis_squared,
                                              cell_weighted_kurtosis_squared, cell_skewness_x, cell_weighted_skewness_x,
                                              cell_skewness_y, cell_weighted_skewness_y, cell_skewness_squared,
                                              cell_weighted_skewness_squared, cell_network_orientation,
                                              cell_network_major_axis,
                                              cell_network_minor_axis, cell_network_eccentricity,
                                              cell_network_effective_extent,
                                              cell_network_effective_solidity, cell_network_fractal_dimension]],
                                            columns=['cell_name', 'cell_mean_mito_area_(pixels_squared)',
                                                     'cell_median_mito_area_(pixels_squared)',
                                                     'cell_std_mito_area_(pixels_squared)',
                                                     'cell_mean_mito_eccentricity',
                                                     'cell_median_mito_eccentricity', 'cell_std_mito_eccentricity',
                                                     'cell_mean_mito_equi_diameter_(pixels)',
                                                     'cell_median_mito_equi_diameter_(pixels)',
                                                     'cell_std_mito_equi_diameter_(pixels)',
                                                     'cell_mean_mito_euler_number',
                                                      'cell_std_mito_euler_number',
                                                     'cell_mean_mito_extent',
                                                     'cell_median_mito_extent', 'cell_std_mito_extent',
                                                     'cell_mean_mito_major_axis_(pixels)',
                                                     'cell_median_mito_major_axis_(pixels)',
                                                     'cell_std_mito_major_axis_(pixels)',
                                                     'cell_mean_mito_minor_axis_(pixels)',
                                                     'cell_median_mito_minor_axis_(pixels)',
                                                     'cell_std_mito_minor_axis_(pixels)',
                                                     'cell_mean_mito_orientation_(degrees)',
                                                     'cell_median_mito_orientation_(degrees)',
                                                     'cell_std_mito_orientation_(degrees)',
                                                     'cell_mean_mito_perimeter_(pixels)',
                                                     'cell_median_mito_perimeter_(pixels)',
                                                     'cell_std_mito_perimeter_(pixels)', 'cell_mean_mito_solidity',
                                                     'cell_median_mito_solidity', 'cell_std_mito_solidity',
                                                     'cell_mean_mito_centroid_x_(pixels)',
                                                     'cell_median_mito_centroid_x_(pixels)',
                                                     'cell_std_mito_centroid_x_(pixels)',
                                                     'cell_mean_mito_centroid_y_(pixels)',
                                                     'cell_median_mito_centroid_y_(pixels)',
                                                     'cell_std_mito_centroid_y_(pixels)',
                                                     'cell_mean_mito_distance_(pixels)',
                                                     'cell_median_mito_distance_(pixels)',
                                                     'cell_std_mito_distance_(pixels)',
                                                     'cell_mean_mito_weighted_cent_x_(pixels)',
                                                     'cell_median_mito_weighted_cent_x_(pixels)',
                                                     'cell_std_mito_weighted_cent_x_(pixels)',
                                                     'cell_mean_mito_weighted_cent_y_(pixels)',
                                                     'cell_median_mito_weighted_cent_y_(pixels)',
                                                     'cell_std_mito_weighted_cent_y_(pixels)',
                                                     'cell_mean_mito_weighted_distance_(pixels)',
                                                     'cell_median_mito_weighted_distance_(pixels)',
                                                     'cell_std_mito_weighted_distance_(pixels)',
                                                     'cell_mean_mito_form_factor', 'cell_median_mito_form_factor',
                                                     'cell_std_mito_form_factor', 'cell_mean_mito_roundness',
                                                     'cell_median_mito_roundness',
                                                     'cell_std_mito_roundness', 'cell_mean_mito_branch_count',
                                                     'cell_std_mito_branch_count',
                                                     'cell_mean_mito_mean_branch_length_(pixels)',
                                                     'cell_median_mito_mean_branch_length_(pixels)',
                                                     'cell_std_mito_mean_branch_length_(pixels)',
                                                     'cell_mean_mito_total_branch_length_(pixels)',
                                                     'cell_median_mito_total_branch_length_(pixels)',
                                                     'cell_std_mito_total_branch_length_(pixels)',
                                                     'cell_mean_mito_median_branch_length_(pixels)',
                                                     'cell_median_mito_median_branch_length_(pixels)',
                                                     'cell_std_mito_median_branch_length_(pixels)',
                                                     'cell_mean_mito_std_branch_length_(pixels)',
                                                     'cell_std_mito_std_branch_length_(degrees)',
                                                     'cell_mean_mito_mean_branch_angle_(degrees)',
                                                     'cell_median_mito_mean_branch_angle_(degrees)',
                                                     'cell_std_mito_mean_branch_angle_(degrees)',
                                                     'cell_mean_mito_median_branch_angle_(degrees)',
                                                     'cell_median_mito_median_branch_angle_(degrees)',
                                                     'cell_std_mito_median_branch_angle_(degrees)',
                                                     'cell_mean_mito_std_branch_angle_(degrees)',
                                                     'cell_std_mito_std_branch_angle_(degrees)',
                                                     'cell_mean_mito_total_density_(pixels)',
                                                     'cell_median_mito_total_density_(pixels)',
                                                     'cell_std_mito_total_density',
                                                     'cell_mean_mito_average_density_(pixels)',
                                                     'cell_median_mito_average_density_(pixels)',
                                                     'cell_std_mito_average_density_(pixels)',
                                                     'cell_mean_mito_median_density_(pixels)',
                                                     'cell_median_mito_median_density',
                                                     'cell_std_mito_median_density_(pixels)', 'cell_kurtosis_x',
                                                     'cell_weighted_kurtosis_x',
                                                     'cell_kurtosis_y', 'cell_weighted_kurtosis_y',
                                                     'cell_kurtosis_squared',
                                                     'cell_weighted_kurtosis_squared', 'cell_skewness_x',
                                                     'cell_weighted_skewness_x',
                                                     'cell_skewness_y', 'cell_weighted_skewness_y',
                                                     'cell_skewness_squared',
                                                     'cell_weighted_skewness_squared',
                                                     'cell_network_orientation_(degrees)',
                                                     'cell_network_major_axis_(pixels)',
                                                     'cell_network_minor_axis_(pixels)', 'cell_network_eccentricity',
                                                     'cell_network_effective_extent', 'cell_network_effective_solidity',
                                                     'cell_network_fractal_dimension'])

                temp_dataset_raw = pd.DataFrame([[file, scale, mito_area, mito_centroid,
                                                  mito_eccentricity, mito_equi_diameter, mito_euler_number, mito_extent,
                                                  mito_major_axis, mito_minor_axis, mito_orientation, mito_perimeter,
                                                  mito_solidity, mito_centroid_x, mito_centroid_y, mito_distance,
                                                  mito_weighted_cent_x, mito_weighted_cent_y, mito_weighted_distance,
                                                  mito_form_factor, mito_roundness, mito_branch_count,
                                                  mito_total_branch_length,
                                                  mito_mean_branch_length, mito_median_branch_length,
                                                  mito_std_branch_length,
                                                  mito_mean_branch_angle, mito_median_branch_angle,
                                                  mito_std_branch_angle,
                                                  mito_total_density, mito_average_density, mito_median_density,
                                                  mito_branch_count,
                                                  mito_distance, mito_weighted_cent_x, mito_weighted_cent_y,
                                                  mito_weighted_distance,
                                                  mito_form_factor, mito_roundness]],
                                                columns=['cell_name', 'resize_factor', 'mito_area', 'mito_centroid',
                                                         'mito_eccentricity',
                                                         'mito_equi_diameter', 'mito_euler_number', 'mito_extent',
                                                         'mito_major_axis',
                                                         'mito_minor_axis', 'mito_orientation', 'mito_perimeter',
                                                         'mito_solidity',
                                                         'mito_centroid_x', 'mito_centroid_y', 'mito_distance',
                                                         'mito_weighted_cent_x',
                                                         'mito_weighted_cent_y', 'mito_weighted_distance',
                                                         'mito_form_factor',
                                                         'mito_roundness', 'mito_branch_count',
                                                         'mito_total_branch_length',
                                                         'mito_mean_branch_length', 'mito_median_branch_length',
                                                         'mito_std_branch_length',
                                                         'mito_mean_branch_angle', 'mito_median_branch_angle',
                                                         'mito_std_branch_angle',
                                                         'mito_total_density', 'mito_average_density',
                                                         'mito_median_density',
                                                         'mito_branch_count', 'mito_distance', 'mito_weighted_cent_x',
                                                         'mito_weighted_cent_y',
                                                         'mito_weighted_distance', 'mito_form_factor',
                                                         'mito_roundness'])

                database = database.append(temp_dataset, ignore_index=True)
                database_raw = database_raw.append(temp_dataset_raw, ignore_index=True)
            except:
                print('Cann\'t test {0}'.format(file))
    database.drop(database.index[0], inplace=True)
    database.to_csv(save_path + '/' + "Distinct image test" + '.csv', sep=',', index=False)

    database_raw.drop(database_raw.index[0], inplace=True)
    database_raw.to_csv(save_path + '/' + "Distinct mitochondria test" + '.tsv', sep='\t', index=False)


    print('Test has been completed')

upload_path = "..\\final_results\\bw"
save_floder = "..\\final_results"
measurement(upload_path, save_floder)
