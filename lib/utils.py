import numpy as np
from scipy import ndimage
from PIL import Image
import os
import cv2
import cPickle as pickle
import matplotlib as mp
#mp.use('Agg')
import matplotlib.image as mpimg
import  matplotlib.pyplot as plt
plt.gray()

repo_gt = '/share/FCNtensorflow/logs/trained_images/ground_truth'
repo_pred = '/share/FCNtensorflow/logs/trained_images/prediction'

visu_gt = "/share/FCNtensorflow/logs/trained_images/ground_truth"
visu_pred = "/share/FCNtensorflow/logs/trained_images/prediction"
visu_water = "/share/FCNtensorflow/logs/trained_images/watershed"

def save_object(obj, filename):
    # Save whole object (used for dataset-readers)
    with open(filename, 'wb') as output:
        pickle.dump(obj, output, pickle.HIGHEST_PROTOCOL)

def load_object(filename):
    # load whole object (used for dataset-readers)
    with open(filename, 'rb') as input:
        return pickle.load(input)

def make_background_black(repository):
    """
    This function just replaces 255 (white) with 0 (black). It is needed because we want the ground truth
    pixels to be on range [0, n] where n is the number of classes
    :param repository: the repository which contains png images
    """
    filenames = os.listdir(repository)
    i = 0
    for file in filenames:
        image = Image.open(os.path.join(repository, file)).convert('L')
        arr = np.asarray(image)
        image.close()
        arr.setflags(write=1)
        arr[arr == 255] = 0
        cv2.imwrite(os.path.join(repository, file), arr)
        if i % 100 == 0:
            print("Processing :" + str(i) + "th image")
        i += 1


def visualize_images(repository_gt, repository_pred):
    """
    This function visualizes images by replacing 0 (black) with 255 (white). It 'undo' make_background_black.
    Call it only for debugging purposes (visualization). The method doesn't save images on white background, it
    just creates temporary ones in order to visualize them.
    :param repository_gt: The repository which contain the ground truth images
    :param repository_pred: The repository which contain the prediction images
    """
    filenames = os.listdir(repository_gt)
    for file in filenames:
        image = Image.open(os.path.join(repository_gt, file))
        arr = np.asarray(image)
        arr.setflags(write=1)
        arr[arr == 0] = 255

        pred = Image.open(os.path.join(repository_pred, "pred" + file[2:]))
        arr2 = np.asarray(pred)
        arr2.setflags(write=1)
        arr2[arr2 == 0] = 255

        plot_image = np.concatenate((arr, arr2), axis=1)

        plt.imshow(plot_image)
        plt.show()


def check_number_of_labels(repository):
    """
    A helping function which checks the number of labels for each png image.
    :param repository: the repository of tiles
    :return: list_of_bad_files - a list with the names of all files that have more labels than the number of classes
    """

    list_of_bad_files = []
    filenames = os.listdir(repository)
    for file in filenames:
        image = Image.open(os.path.join(repository, file))
        arr = np.asarray(image)
        # find the unique values
        unique = np.unique(arr)
        control = 0
        for num in unique:
            if num > 9:
                control = 1
        if control:
            list_of_bad_files.append(file)
    return list_of_bad_files


def compare_images_visually(repo_of_images_gt, repo_of_images_pred, ignore_background=True):
    """
    This function compares images visually, by putting green pixels where pixels' value match, and red values where
    they don't match
    :param repository_gt: The repository which contain the ground truth images
    :param repository_pred: The repository which contain the prediction images
    :param ignore_background: Parameter which determines if the algorithm ignores background
    """
    filenames = os.listdir(repo_of_images_gt)
    for file in filenames:
        image_gt = Image.open(os.path.join(repo_of_images_gt, file))
        image_pred = Image.open(os.path.join(repo_of_images_pred, "pred" + file[2:]))
        arr_gt = np.asarray(image_gt)
        arr_pred = np.asarray(image_pred)
        height, width = arr_gt.shape
        arr_compare = np.ones((arr_gt.shape[0], arr_gt.shape[1], 3))  # we want the comparison image in rgb
        for i in xrange(height):
            for j in xrange(width):
                if arr_gt[i, j] == arr_pred[i, j] and arr_gt[i, j] != 0:
                    arr_compare[i, j] = (0, 1, 0)
                elif arr_gt[i, j] != arr_pred[i, j]:
                    arr_compare[i, j] = (1, 0, 0)
        plt.imshow(arr_compare)
        plt.show()


def compute_mean_average_precision(repo_of_images_gt, repo_of_images_pred, ignore_background=True):
    """
    This function computes the mean average precision
    :param repository_gt: The repository which contain the ground truth images
    :param repository_pred: The repository which contain the prediction images
    :param ignore_background: Parameter which determines if the algorithm ignores background
    """
    filenames = os.listdir(repo_of_images_gt)
    average_precision = []
    for file in filenames:
        image_gt = Image.open(os.path.join(repo_of_images_gt, file))
        image_pred = Image.open(os.path.join(repo_of_images_pred, "pred" + file[2:]))
        arr_gt = np.asarray(image_gt)
        arr_pred = np.asarray(image_pred)
        height, width = arr_gt.shape
        num_matches, num_mismatches = 0.0, 0.0
        for i in xrange(height):
            for j in xrange(width):
                if ignore_background: # check if we have to ignore the background
                    if arr_gt[i, j] != 0 or arr_pred[i, j] != 0:  # check if it is a background pixel
                        if arr_gt[i, j] == arr_pred[i, j]:
                            num_matches += 1
                        else:

                            num_mismatches += 1
                else:
                    if arr_gt[i, j] == arr_pred[i, j]:
                        num_matches += 1
                    else:
                        num_mismatches += 1

        if num_matches + num_mismatches != 0:
            print("Match precision is: " + str(num_matches/(num_matches + num_mismatches)))
            average_precision.append(num_matches/(num_matches + num_mismatches))
    average_precision = np.asarray(average_precision)
    print("Mean average precision is: " + str(np.mean(average_precision)))

def do_wathershed(visu_gt, visu_pred):
    print("no implementare yet")
    # remove stuff outside of objects
    vis_pred = os.listdir(visu_pred)
    for file in vis_pred:
        if "m" in file:
            #show class labels
            pred_m = Image.open(os.path.join(visu_pred, file))
            arr_pred_m = np.asarray(pred_m)
            arr_pred_m.setflags(write=1)
            arr_pred_m[arr_pred_m == 0] = 255
            arr_pred_m = arr_pred_m + (arr_pred_m *15)*(arr_pred_m!=255)
            img_sm = Image.fromarray(arr_pred_m, 'L')
            img_sm.show(title="pred")

            pred_o = Image.open(os.path.join(visu_pred, file[:-5] + "o.png"))
            arr_pred_o = np.asarray(pred_o)
            arr_pred_o.setflags(write=1)
            arr_pred_o[arr_pred_o == 0] = 255
            arr_pred_o = arr_pred_o + (arr_pred_o *15)*(arr_pred_o!=255)
            arr_pred_o[0:3,:] = 1
            img_sm = Image.fromarray(arr_pred_o, 'L')
            img_sm.show(title="pred")



            #smoothen objects
            pred_o = np.asarray(pred_o)
            pred_o = pred_o.astype(np.float32)
            pred_o[pred_o < 0.1] = 255
            pred_f_o = ndimage.filters.gaussian_filter(pred_o,sigma=[0.2,0.2])
            print("smooth")
            pred_f_o = pred_f_o + (pred_f_o * 15) * (pred_f_o != 255)
            pred_f_o = pred_f_o.astype(np.int8)
            img_sm = Image.fromarray(pred_f_o, 'L')
            img_sm.show(title="smooth_pred")
            # img.save('my.png')

            #look at gt
            pred_o = Image.open(os.path.join(visu_gt, "gt"+(file[:-5] + "o.png")[4:]))
            arr_pred_o = np.asarray(pred_o)
            arr_pred_o.setflags(write=1)
            arr_pred_o = arr_pred_o-1
            arr_pred_o[arr_pred_o<0]==0
            arr_pred_o[arr_pred_o == 0] = 255
            arr_pred_o = arr_pred_o + (arr_pred_o * 15) * (arr_pred_o != 255)

            img_sm = Image.fromarray(arr_pred_o, 'L')
            img_sm.show(title="gt")


    # plt.imshow(pred_annotations_o[0])
    # plt.show()





    return None


def vis_predicitons(visu_gt, visu_pred):
    vis_gt = os.listdir(visu_gt)
    for file in vis_gt:
        if "m" in file:
            gt_image = Image.open(os.path.join(visu_gt, file))
            arr_gt = np.asarray(gt_image)
            arr_gt.setflags(write=1)
            gt_image = Image.open(os.path.join(visu_pred, "pred"+file[2:]))
            arr_pred = np.asarray(gt_image)
            arr_pred.setflags(write=1)


            shape = arr_gt.shape
            # white canvas
            data = np.ones((shape[0], shape[1], 3), dtype=np.uint8)*255

            # correct but not bg --> green
            data[(arr_gt == arr_pred) * (arr_pred != 0), :] =  [0,255,0]

            # wrong and none of the parts are bg --> red
            sel = (arr_gt != arr_pred)*(arr_pred>0)*(arr_gt>0)
            data[sel,:] = [255,0,0]

            # wrong but involves bg --> blue
            sel = (arr_gt != arr_pred)*((arr_pred==0)+(arr_gt==0))
            data[sel,:] = [0,0,255]


            img = Image.fromarray(data, 'RGB')
            #img.save('my.png')
            img.show()

            # switch name to objectiveness

            file = file[:-5]+"o.png"

            gt_image = Image.open(os.path.join(visu_gt, file))
            arr_gt = np.asarray(gt_image)
            arr_gt.setflags(write=1)
            gt_image = Image.open(os.path.join(visu_pred, "pred"+file[2:]))
            arr_pred = np.asarray(gt_image)
            arr_pred.setflags(write=1)

            shape = arr_gt.shape
            # white canvas
            data = np.ones((shape[0], shape[1], 3), dtype=np.uint8) * 255

            # correct but not bg --> green
            data[(arr_gt == arr_pred) * (arr_pred != 0), :] = [0,255,0]

            # wrongly set as foreground--> dark_red
            sel = (arr_gt != arr_pred)*(arr_gt==0)
            data[sel,:] = [255,0,255]

            # wrongly set as background --> red
            sel = (arr_gt != arr_pred)*(arr_pred==0)
            data[sel,:] = [255,0,0]

            # low engergy --> blue
            sel = (arr_gt > arr_pred)*(arr_pred>0)*(arr_gt>0)
            data[sel,:] = [0,0,255]

            # high engergy -->
            sel = (arr_gt < arr_pred)*(arr_pred>0)*(arr_gt>0)
            data[sel,:] = [0,255,255]

            img = Image.fromarray(data, 'RGB')
            #img.save('my.png')
            img.show()


if __name__ == "__main__":
    #do_wathershed(visu_gt, visu_pred)
    vis_predicitons(visu_gt, visu_pred)
    # compare_images_visually(repo_gt, repo_pred)
    # compute_mean_average_precision(repo_gt, repo_pred, 1)
    # 0.615384665233 on 300k iterations (batch_size = 2) ~37  epochs
    # 0.634142536941 on 1m  c iterations (batch_size = 2) ~125 epochs


print("Done")