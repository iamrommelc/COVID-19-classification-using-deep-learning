import os
from PIL import Image

i=0
for image_file_name in os.listdir(source_imagefolder_path):
    if image_file_name.endswith(<imagefile_extension>):
        i=i+1
        im = Image.open(<source_imagefile_path>+image_folder_name)
        new_w=224
        new_h=224
        im = im.resize((new_w,new_h), Image.ANTIALIAS)
        im.save(<destination_imagefolder_path>+ str(i) + '.png')
