# Imports
import os, argparse, random, shutil, math

# Parses Terminal Inputs
def get_inputs():
    parser = argparse.ArgumentParser(description='Creates random splits and saves them to a txt file',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--val-size',type=str,required=True,
                    help='How many pictures you want in the validation set, or just `all` to use everything in the folder')
    parser.add_argument('--test-size', type=str,required=True,
                    help='How many pictures you want in the test set, or just `all` to use everything else in the folder')
    parser.add_argument('--train-size', type=str, required=True,
                    help='How many pictures you want in the training set, or just `all` to use everything in the folder')

    parser.add_argument('--test-dir', type=str, required=True,
                    help='The directory to the folder wanted for the test set')
    parser.add_argument('--val-dir', type=str, required=True,
                    help='The directory to the folder wanted for the validation set')
    parser.add_argument('--train-dir', type=str, required=True,
                    help='The directory to the folder wanted for the training set')

    parser.add_argument('--save-dir', type=str, required=True,
                    help='Where to the save the splits too (the name of the last folder should be one that doesnt exist)')

    return parser.parse_args()

opt = get_inputs()

# Variables holding specific data about the training, val, and test sets
set_type = ('training','validation','test')
set_type_directory = (opt.train_dir,opt.val_dir,opt.test_dir)

set_size = [int(opt.train_size) if opt.train_size != 'all' else opt.train_size,
            int(opt.val_size) if opt.val_size != 'all' else opt.val_size,
            int(opt.test_size) if opt.test_size != 'all' else opt.test_size]

file_sets = {key: [] for key in set_type}

# Randomly picks images for the test, train, and val sets
def random_split():
    global set_size
    
    #Checks if an image is already in a set
    def check_files(index,file_check):
        if index == 0: return True
        for split in range(index):
            for check in file_sets[set_type[split]]:
                if file_check in file_sets[set_type[split]]:
                    return False
        return True

    # Loops through each type of split
    for split_type in range(len(set_type)):
        # Loops through each folder
        for folder in os.scandir(set_type_directory[split_type]):
            folder = os.path.abspath(folder)
            
            # Checks if the file is a folder, and will move on if it is not
            try:
                os.scandir(folder)
            except NotADirectoryError:
                continue
            else:
                # Creates a list holding all the image directories for each folder
                files = [os.path.abspath(path) for path in os.scandir(folder)]

                # Checks if the user wants the images to be used
                if set_size[split_type] == 'all' or set_size[split_type] > len(files):
                    for size in range(len(set_size)):
                        set_size[split_type] = len(files)

                # Checks if there is enough images for the specifications, and if not, splits it 
                if set_type_directory[0] == set_type_directory[1] and set_size[0]+set_size[1] > len(files) and split_type < 2:
                    set_size[0] = math.floor(len(folder)*0.8)
                    set_size[1] = math.floor(len(folder)*0.2)
                    print('here')
                    print (f'{folder} does not have enough files for the training and val sets, so there'+
                           'will be a {set_size[0]} (training images) {set_size[1]} (val images) split.')
                
                # Adds randomly picked files
                picked_files = 0
                while picked_files < set_size[split_type]:
                    # Picks a file
                    file = random.choice(files)

                    # Checks the extension
                    name, ext = os.path.splitext(os.path.basename(file))

                    # Adds the file to a set if it is not in a another set and is an image
                    if check_files(split_type,file) and ext.lower() in ['.jpeg','.jpg','.png',]:
                        file_sets[set_type[split_type]].append(file)
                        picked_files += 1

                    # Removes the file from the selection
                    files.remove(file)
            
            # Resets the training and val to the original input
            set_size = [int(opt.train_size) if opt.train_size != 'all' else opt.train_size,
                        int(opt.val_size) if opt.val_size != 'all' else opt.val_size,
                        int(opt.test_size) if opt.test_size != 'all' else opt.test_size]
# Creates the split folder and sub folders, then moves the images over
def create_splits():
    # Loops through all splits
    for split_type in set_type:

        # Grabs the name of the folder that the original image is stored in
        labels = []
        for name in file_sets[split_type]:
            name = os.path.basename(os.path.dirname(name))
            # Checks if the name is already put in or not
            if not name in labels:
                labels.append(name)

        # Creates the path for the save directory
        save_path = opt.save_dir+'/'+split_type+'/'

        # Loops through the folder names grabbed earlier
        for l in labels:
            # Attempts to make the directory, and does nothing if it already is
            try:
                os.makedirs(save_path+l)
            except FileExistsError:
                None

        # Makes a copy of the images and places them into the newly made dataplit folders
        for file in file_sets[split_type]:
            shutil.copy(file,save_path+os.path.basename(os.path.dirname(file)))

# Calls the functions defined
random_split()
create_splits()
