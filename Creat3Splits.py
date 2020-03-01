import split_folders

# Split with a ratio.
# To only split into training and validation set, set a tuple to `ratio`, i.e, `(.8, .2)`.

input_folder = 'center_crop_clean_300\\'
output_folder = 'data_3C_from_center_crop_clean_300_to_300'
split_folders.ratio(input_folder, output=output_folder, seed=1337, ratio=(.7, .2, .1)) # default values
