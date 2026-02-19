import os

for root, dirs, files in os.walk('/mimer/NOBACKUP/groups/alvis_cvl/datasets/ImageNet_2012'):
    if root == '/mimer/NOBACKUP/groups/alvis_cvl/datasets/ImageNet_2012/train':
        print('Train folders: ', len(dirs))
    if root == '/mimer/NOBACKUP/groups/alvis_cvl/datasets/ImageNet_2012/val':
        print('Val folders: ', len(dirs))