import cv2
import os
import sys
import numpy as np
import pickle
from absl import app
from absl import flags

FLAGS = flags.FLAGS
flags.DEFINE_string("dataset", "./tumiitgait/Static Occlusion", "Location of folder to annote")
flags.DEFINE_integer("height", 160, "Resized height of the cropped image")
flags.DEFINE_integer("width", 120, "Resized width of the cropped image")
flags.DEFINE_boolean("autocrop", False, "True if autocropping to be done inside the annotated bounding box")

def preprocess(img,typeinp, FLAGS):
    h ,w = 0, 0
    if typeinp=="complete":
        kernel = np.ones((5,5), np.uint8)
        dilation = cv2.dilate(img, kernel, iterations=1)
        img = cv2.erode(dilation, kernel, iterations=1)
        cnt, heir = cv2.findContours(img[:,:,0],cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        cnt = sorted(cnt, key=cv2.contourArea)[::-1]
        if (len(cnt)>0):
            x,y,w,h = cv2.boundingRect(cnt[0])
            img = img[y:y+h, x:x+w, 0]
            
        else:
            h = img.shape[0]
            w = img.shape[1]
    else:
        h = img.shape[0]
        w = img.shape[1]
            #img = np.zeros((FLAGS.height,FLAGS.width))
    #print("heigth", h,"width", w)
    w1 = (h*3)//4
    img = cv2.copyMakeBorder(img, 0,0, max(w1-w,0)//2, max(w1-w,0)//2, cv2.BORDER_CONSTANT, (0,0,0))
    img= cv2.resize(img, (FLAGS.width,FLAGS.height))
    
    return img

def main(argv):
    assert os.path.isdir(FLAGS.dataset)
    for vid_file in sorted(os.listdir(FLAGS.dataset)):
        if not (vid_file.find('avi')>=0 or vid_file.find('mp4')>=0):
            continue
        print(vid_file)
        info = {}
        subject = vid_file.split('.')[0]
        if not os.path.isdir('/'.join([FLAGS.dataset, subject])):
            os.mkdir('/'.join([FLAGS.dataset, subject]))
        cap = cv2.VideoCapture( '/'.join([FLAGS.dataset, vid_file]) )
        ret, nxt = cap.read()
        cnt = 1
        file = str(cnt).zfill(6) + ".png"
        while ret:
            r = cv2.selectROI(nxt)
            if r[2]>0 and r[3]>0:
                crp_img = nxt[int(r[1]):int(r[1]+r[3]), int(r[0]):int(r[0]+r[2])]
                cv2.imshow("frame",crp_img)
                k = cv2.waitKey(0)&0xFF

                ord('o') ## occluded
                ord('c') ## complete
                ord('n') ## no subject
                if k == ord('q'):
                    break
                elif k == ord('o'):
                    info[file] = "occluded"
                    crp_img = preprocess(crp_img, info[file], FLAGS)
                elif k == ord('c'):
                    info[file] = "complete"
                    crp_img = preprocess(crp_img, info[file], FLAGS)
                elif k == ord('n'):
                    info[file] = "no_subject"
                else:
                    info[file] = "no_subject"

                
                if info[file]!="no_subject":
                    cv2.imshow("frame",crp_img)
                    k = cv2.waitKey(0)&0xF
                    if k==ord('q'):
                        break
                    elif k!=ord('r'):
                        cv2.imwrite('/'.join([FLAGS.dataset, subject, file]), crp_img)
                    
                cnt += 1
                file = str(cnt).zfill(6) + ".png"
                cv2.destroyWindow('frame')
            ret, nxt = cap.read()
              
        with open('/'.join([FLAGS.dataset, subject, 'info.pkl'])):
            pickle.dump(info, handle, protocol=pickle.HIGHEST_PROTOCOL)

if __name__=="__main__":
    app.run(main)