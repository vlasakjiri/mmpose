import cv2
from glob import glob

images = glob('fig/*.jpg')

for img_path in images:
    img = cv2.imread(img_path)

    # Crop image
    x, y, w, h = 600, 0, 650, 950
    crop_img = img[y:y+h, x:x+w]

    # Save image
    basename = img_path.split('/')[-1].split('.')[0]

    cv2.imwrite(basename + "_crop" +".jpg", crop_img)
