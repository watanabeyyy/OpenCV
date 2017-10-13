########################################################
## Title  : Gazou classification of alpha terminal
## Author : Moriki Kazuya / Watanabe yuki
## Date   : 2017.09.25
########################################################
import glob
import shutil
import cv2
  # define the dir of data
data_dir_path = '../img/'
  # define the name and extension
image_list = glob.glob(data_dir_path + '*.jpg')
  # make the lists
OK = []
NG = []
iranai = []
  # Process each images
for i, path in enumerate(image_list):
    image = cv2.imread(path)
    image = cv2.resize(image, (500, 500))
    cv2.imshow('image', image)
    # [windows]OK   is 1 key: 49
    # [windows]iranai   is 2 key: 50
    # [windows]NG   is 3 key: 51
    # [LINUX]OK     is 6 key: 1114038
    # [LINUX]iranai is 5 key: 1114037
    # [LINUX]NG     is 4 key: 1114036
    key = 0
    # define the number of Key for OS
    while key != 49 and key != 50 and key != 51:
        key = cv2.waitKey(0)
        print(key)
    if key == 49:
        OK.append(path)
        print('OK')
        shutil.move(path, '../outputs/OK/.')
    elif key == 50:
        NG.append(path)
        print('iranai')
        shutil.move(path, '../outputs/iranai/.')
    elif key == 51:
        NG.append(path)
        print('NG')
        shutil.move(path, '../outputs/NG/.')
    cv2.destroyAllWindows()