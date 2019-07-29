# Imports
import time
import argparse
import json
import math
import os

from gluoncv.data import ImageNet1kAttr
from gluoncv.data.transforms.presets.imagenet import transform_eval
from gluoncv.model_zoo import get_model
from mxnet import nd, image


# Takes a string and makes it a boolean
def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


# Terminal command inputs
def get_args():
    parser = argparse.ArgumentParser(description='Predict ImageNet classes for inputed images')
    parser.add_argument('--model', type=str, required=True,
                        help='name of the model to use')

    parser.add_argument('--saved-params', type=str, default='',
                        help='path to the saved model parameters')

    parser.add_argument('--input-fldr', type=str, required=True,
                        help='path to the input picture')

    parser.add_argument('--extra-fldrs', type=str2bool, default=False,
                        help='Make `True` if you want subfolders to be searched')

    parser.add_argument('--save-to-file', type=str2bool, default=False,
                        help='Set to `True` to save the results to a json file')

    parser.add_argument('--display-in-terminal', type=str2bool, default=True,
                        help='Set to `False` to have it not displayed in the console')

    parser.add_argument('--top-k', type=int, default=5,
                        help='The number of rankings you want displayed')

    return parser.parse_args()


opt = get_args()

# Takes in what directory the user wants the outputs saved too, if wanted
directory = input('Choose a directory to save the file too (include final file, but no extension): ') \
    if opt.save_to_file else None

# Load Model
model_name = opt.model
pretrained = True if opt.saved_params == '' else False
net = get_model(model_name, pretrained=pretrained)

if not pretrained:
    net.load_parameters(opt.saved_params)
    attrib = ImageNet1kAttr()
    classes = attrib.classes
else:
    classes = net.classes

# Stores the saved picture directories
pictures = []


# Recurses through folders
def find_all_images(input_file=None):
    # Attempts scan the file provided, seeing if it is a folder
    for file in os.scandir(opt.input_fldr if input_file is None else input_file):
        file = os.path.abspath(file)

        try:
            # Checks if the input is a folder
            os.scandir(file)

        # If the file input isn't a folder, then it adds it the picture array
        except Exception:
            # Grabs the non-folder file extension and checks if it is one of the images types listed
            name, ext = os.path.splitext(os.path.basename(file))
            if ext in ('.JPEG', '.jpg', '.png', '.jpeg'):
                pictures.append(file)

        # If the input is a folder, then run this same function on the newly found folder
        else:
            find_all_images(file)


start_time = time.time()


# Predicts the images
def pred_images():
    # Get the image directories
    find_all_images()

    # Dictionary that stores all the information to be saved to a file
    save_data = {}

    # Loops through all images
    for images in pictures:
        # Updates to the right directory
        os.chdir(os.path.dirname(images))

        # Create the dictionary to hold all the classes and confidences
        save_data[images] = {}

        # Load Images
        img = image.imread(os.path.basename(images))

        # Transform
        img = transform_eval(img)

        # Prediction
        pred = net(img)

        # Prints the prediction
        ind = nd.topk(pred, k=opt.top_k)[0].astype('int')

        if opt.display_in_terminal:
            for i in range(opt.top_k):
                pred_obj = classes[ind[i].asscalar()]
                pred_score = nd.softmax(pred)[0][ind[i]].asscalar()

                if i == 0:
                    print(f'\n{os.path.basename(images)} is classified to be:')

                print(f'\t{pred_obj}, with probability {math.floor(pred_score * 1000) / 1000}')

        # Now goes through through each class and gets the confidence
        ind = nd.topk(pred, k=1000)[0].astype('int')

        for simple in range(1000):
            pred_obj = classes[ind[simple].asscalar()].replace(' ', '_')
            pred_score = nd.softmax(pred)[0][ind[simple]].asscalar()

            save_data[images][pred_obj] = pred_score

    # Saves the results to a json file (if requested to do so)
    if opt.save_to_file:
        try:
            os.mkdir(os.path.dirname(directory))
        except FileExistsError:
            # Nothing will happen if the file exists because it will simply put it in that directory
            pass

        # Sets the proper directory
        os.chdir(os.path.dirname(directory))

        # Saves the data
        with open(os.path.basename(directory) + ".json", "w") as file_obj:
            json.dump(save_data, file_obj, indent=2, sort_keys=True)

    endtime = time.time()
    print(f'{endtime - startime}')


pred_images()
