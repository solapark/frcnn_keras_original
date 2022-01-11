import cv2
from utils import get_list_from_csv

label_info_path = '/data3/sap/EPFL_MVMC/instance_label.csv'
image_base_path = '/data3/sap/EPFL_MVMC/image_instance/c%d/%08d.jpg'
color_list = [(0, 0, 255), (255, 0, 0), (0, 255, 0), (255, 255, 255), (0, 0, 0)]

#for line : read img & draw & save
if __name__ == '__main__' :
    lines = get_list_from_csv(label_info_path, header=False)
    for line in lines :
        frame, cam, cls, _, x1, y1, x2, y2, inst = line
        image_path = image_base_path %(int(cam), int(frame))
        image = cv2.imread(image_path)
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

        color = color_list[int(inst)%len(color_list)]
        text_point = (x1-10, y1-10)
        cv2.putText(image, inst, text_point, 1, 2, color, 2)
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 1)
        #cv2.imshow('image', image)
        #cv2.waitKey()
        cv2.imwrite(image_path, image)
   
