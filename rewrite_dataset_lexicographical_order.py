import os
import glob


import re

def rewrite_dataset_frame_order(path):
    """ renames a list of files from normal numerical order to lexicographical.
        This function is file pattern specific and is not supported for multiple use. Change the code according to file pattern input 

    """
    
    file_extension = "png"


    if not os.path.isdir(path):
        raise ValueError('Provided path %s is not a directory' % path)
    else:
        im_list = sorted(glob.glob(os.path.join(path, '*.%s' % file_extension)))

    numbers_str = [re.search(r'_(\d+)\D*$', filename).group(1) for filename in im_list]
    numbers_int = [int(re.search(r'_(\d+)\D*$', filename).group(1)) for filename in im_list]

    print(im_list)

    max_digits = len(str(max(numbers_int)))

    pad = "0" * max_digits

    numbers_lexicographical = [pad[0:3-len(num_val)] + num_val for num_val in numbers_str]

    for i, name in enumerate(im_list):

        prefix = re.match(r'^(.*_)', name).group(1)

        name_lexicographicap = prefix + numbers_lexicographical[i] + "." + file_extension


        os.rename(name, name_lexicographicap)


def list_of_videos_2_lexicographical_order(data_frames_path):
    # data_frames_path = "/home/petek/robwork_run_directory/amps_0_200_dataset_stuck_on_track"

    for names in os.listdir(data_frames_path):
        videos_path = os.path.join(data_frames_path, names)
        if os.path.isdir(videos_path):
            rewrite_dataset_frame_order(videos_path)





if __name__ == '__main__':

    data_frames_path = "data/new_normal_data_DCM4"
    list_of_videos_2_lexicographical_order(data_frames_path=data_frames_path)
