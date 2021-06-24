import cv2
import numpy as np
from skimage import transform as trans


def convert_68pt_to_5pt(landmarks):
	left_eye_x = round((landmarks[43][0] + landmarks[44][0] + landmarks[46][0] + landmarks[47][0])/4.0)
	left_eye_y = round((landmarks[43][1] + landmarks[44][1] + landmarks[46][1] + landmarks[47][1])/4.0)
	
	right_eye_x = round((landmarks[37][0] + landmarks[38][0] + landmarks[40][0] + landmarks[41][0])/4.0)
	right_eye_y = round((landmarks[37][1] + landmarks[38][1] + landmarks[40][1] + landmarks[41][1])/4.0)
	
	nose_x = round((landmarks[30][0] + landmarks[33][0])/2.0)
	nose_y = round((landmarks[30][1] + landmarks[33][1])/2.0)
	
	nose_x = round((landmarks[30][0]))
	nose_y = round((landmarks[30][1]))            
	
# 	mapped_landmarks = [[left_eye_x, left_eye_y], [right_eye_x, right_eye_y], [nose_x, nose_y], [landmarks[54][0], landmarks[54][1]], [landmarks[48][0], landmarks[48][1]]]
	mapped_landmarks = [[right_eye_x, right_eye_y], [left_eye_x, left_eye_y], [nose_x, nose_y], [landmarks[48][0], landmarks[48][1]], [landmarks[54][0], landmarks[54][1]]]
# 	mapped_landmarks.shape = -1, 5, 2
	return np.array(mapped_landmarks) 

def face_recover(img, M, ori_img, landmark):
    # img:rgb, ori_img:bgr
    # dst:rgb
    dst = ori_img.copy()
    inv_trans = np.linalg.pinv(M)
    cv2.warpAffine(img, M, (dst.shape[1], dst.shape[0]), dst,
                   flags=cv2.WARP_INVERSE_MAP | cv2.INTER_CUBIC , borderMode=cv2.BORDER_TRANSPARENT)
    return dst

def dlib_alignment(img, landmarks, padding=0.0, size=96, moving=0.0):
    x_src = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.000213256, 0.0752622, 0.18113, 0.29077, 0.393397, 0.586856, 0.689483, 0.799124,
                      0.904991, 0.98004, 0.490127, 0.490127, 0.490127, 0.490127, 0.36688, 0.426036,
                      0.490127, 0.554217, 0.613373, 0.121737, 0.187122, 0.265825, 0.334606, 0.260918,
                      0.182743, 0.645647, 0.714428, 0.793132, 0.858516, 0.79751, 0.719335, 0.254149,
                      0.340985, 0.428858, 0.490127, 0.551395, 0.639268, 0.726104, 0.642159, 0.556721,
                      0.490127, 0.423532, 0.338094, 0.290379, 0.428096, 0.490127, 0.552157, 0.689874,
                      0.553364, 0.490127, 0.42689])
    y_src = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.106454, 0.038915, 0.0187482, 0.0344891, 0.0773906, 0.0773906, 0.0344891,
                      0.0187482, 0.038915, 0.106454, 0.203352, 0.307009, 0.409805, 0.515625, 0.587326,
                      0.609345, 0.628106, 0.609345, 0.587326, 0.216423, 0.178758, 0.179852, 0.231733,
                      0.245099, 0.244077, 0.231733, 0.179852, 0.178758, 0.216423, 0.244077, 0.245099,
                      0.780233, 0.745405, 0.727388, 0.742578, 0.727388, 0.745405, 0.780233, 0.864805,
                      0.902192, 0.909281, 0.902192, 0.864805, 0.784792, 0.778746, 0.785343, 0.778746,
                      0.784792, 0.824182, 0.831803, 0.824182])


    x_src = (padding + x_src) / (2 * padding + 1)
    y_src = (padding + y_src) / (2 * padding + 1)
    y_src += moving
    x_src *= size
    y_src *= size
    src = np.concatenate([np.expand_dims(x_src, -1), np.expand_dims(y_src, -1)], -1)
    
    dst = landmarks.astype(np.float32)
    src = convert_68pt_to_5pt(src)
    dst = convert_68pt_to_5pt(dst)
    # print("SRC LANDMARKS:")
    # print(src)
    # print("DST LANDMARKS:")
    # print(dst)
    
    tform = trans.SimilarityTransform()
    tform.estimate(dst, src)
    M = tform.params[0:2, :]

    warped = cv2.warpAffine(img, M, (size, size), flags=cv2.INTER_AREA, borderValue=0.0)

    return warped, M