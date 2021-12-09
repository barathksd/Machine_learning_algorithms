import numpy as np
#import argparse
#import imutils
import cv2
import matplotlib.pyplot as plt
import pandas as pd
import csv
import json
import imutils
from scipy.optimize import leastsq
from scipy.optimize import linear_sum_assignment
#import pymysql
#import paramiko
#import face_recognition
import time
from datetime import datetime
import datetime as dtlib
import base64 
from PIL import Image
from io import BytesIO

def imgtob64(img):
    #with open(base+'\\images\\1.jpg', "rb") as image_file:
    #    data = base64.b64encode(image_file.read())
    
    #print(type(data))
    #with open(base+"\\images\\datab", "wb") as txtfile:
    #    txtfile.write(base64.b64decode(data))
    
    #with open(base+'\\images\\datab', "rb") as image_file:
    #    data2 = base64.b64encode(image_file.read())
    #print(data==data2)
    #im = Image.open(BytesIO(base64.b64decode(data)))
    
    retval, buffer = cv2.imencode('.jpg', img)
    return base64.b64encode(buffer)

def b64toimg(b64img):
    imgio = BytesIO()
    imgio.write(base64.b64decode(b64img))
    return cv2.cvtColor(np.array(Image.open(imgio)),cv2.COLOR_BGR2RGB)



#from sklearn.preprocessing import PolynomialFeatures
def totlbr(boxs):
    boxs[:,2] = boxs[:,2] + boxs[:,0]
    boxs[:,3] = boxs[:,3] + boxs[:,1]
    return boxs
    
def toxywh(boxs):
    boxs[:,2] = boxs[:,2] - boxs[:,0]
    boxs[:,3] = boxs[:,3] - boxs[:,1]
    return boxs
    
def mouse(img,file=None):
    points = []
    
    class counter:
        val = 0
        
    def mousepoint(event,x,y,flags,param): 
        if event == cv2.EVENT_LBUTTONDOWN:
            print('-',x,y,img[y,x],'  --------')
            points.append((x,y))
            cv2.circle(img,(x,y),10,(255,0,0),2)
            cv2.putText(img, str(counter.val),(x,y), cv2.FONT_HERSHEY_SIMPLEX,  
                   1, (0,0,255), 1, cv2.LINE_AA)
            counter.val += 1
            
    def ctrl(img,points):
    
        while(True):
            cv2.namedWindow('image')
            cv2.setMouseCallback('image',mousepoint)

            while(True):
                cv2.imshow('image',img)
                if cv2.waitKey(0) == 27:
                    break
            cv2.destroyAllWindows()
            break

    ctrl(img,points)
    points = pd.DataFrame(points,columns=['x','y'])
    if file:
        points.to_csv(file)
    #cv2.imwrite(base+'\\images\\cam2.jpg',img)
    return img,points

    
def imgplot(img,cmap='gray'):
    img = img.copy()
    fig = plt.figure()
    
    if np.sum(img.shape)>1000:
        fig = plt.gcf()
        fig.set_size_inches(24,12)
        
    else:
        fig.set_size_inches(6,4)
        
    if len(img.shape) == 3:
        plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    else:
        plt.imshow(img,cmap)
    plt.show()


# display picture in openCV
def dispcv(img,imgl=None):
    
    cv2.imshow('img',img)
    if not imgl is None:
        n = len(imgl)
        for i in range(n):
            if not imgl[i] is None:
                cv2.imshow('img'+str(i),imgl[i])
        
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def hilo(a, b, c):
    if c < b: b, c = c, b
    if b < a: a, b = b, a
    if c < b: b, c = c, b
    return a + c

def complement(r, g, b):
    k = hilo(r, g, b)
    return tuple(k - u for u in (r, g, b))

def colorboxes(image,boxes,btype = 'tlbr',color = (120,60,180),text=None,tcolor=(120,250,100)):
    
    if tcolor is None:
        tcolor = complement(*color)
        
    if len(boxes)==0:
        return image
    
    for box in boxes:
        image = colorbox(image,box[1:],btype,color,box[0],tcolor)
    return image

def colorbox(image,box,btype = 'tlbr',color = (120,60,180),text=None,tcolor=(120,250,100)):
    
    if tcolor is None:
        tcolor = complement(*color)
    
    (x, y) = (int(box[0]), int(box[1]))
    
    if btype == 'tlbr':
        (w, h) = (int(box[2]-x),int(box[3]-y))
        
    elif btype == 'tlwh':
        (w, h) = (int(box[2]),int(box[3]))
    
    elif btype == 'xywh':
        (w, h) = (int(box[2]),int(box[3]))
        x,y = (x-int(w/2), y-int(h/2))
        
    cv2.rectangle(image, pt1=(x, y-15), pt2=(x+w, y), color = color, thickness = -1)
    cv2.rectangle(image, pt1=(x, y), pt2=(x+w, y+h), color = color, thickness = 2)
    
    if text != None:
        text = str(text)
        cv2.putText(image, text, (int(x), int(y) - 5), cv2.FONT_HERSHEY_SIMPLEX,
                0.5, tcolor, 2)
    return image

def getvs(vidlink):
    vs = cv2.VideoCapture(vidlink)
    
    try:
        prop = cv2.cv.CV_CAP_PROP_FRAME_COUNT if imutils.is_cv2() \
            else cv2.CAP_PROP_FRAME_COUNT
        total = int(vs.get(prop))
        print("[INFO] {} total frames in video".format(total),vidlink)
    except:
        print("[INFO] could not determine # of frames in video")
        total = -1
        
    return vs,total

def getfrfaces(image,doplot=False,q=None):
    import face_recognition
    fboxes = face_recognition.face_locations(image, model="cnn")
    #encodings = face_recognition.face_encodings(image,fboxes,model='cnn')
    if len(fboxes) == 0:
        return False
    
    fboxes = pd.DataFrame([[fl[3],fl[0],fl[1],fl[2]] for fl in fboxes],columns=['tx','ty','bx','by'])  # top left and bottom right
    fboxes['index'] = np.arange(fboxes.shape[0])
    
    if doplot == True:
        color=(240,60,120)
        for fl in fboxes.iterrows():
            x,y,x2,y2,_ = fl[1]
            cv2.rectangle(image, (x, y), (x2, y2), color, 2)
        imgplot(image)
    
    if q:
        q.put(fboxes)
        return
    return fboxes

def getboth(path,net,image=None,reshape=False,doplot=False):
    
    if type(image) == type(None):
        image = cv2.imread('C:\\Users\\81807\\Documents\\RD\pics\\street2.jpg')
    if reshape:
        image = reshapeimg(image)
    
    pos = givebox('',path,net,image.copy(),0,doplot=False)
    boxes = np.array([(index,p['x'],p['y'],p['x']+p['w'],p['y']+p['h']) for index,p in enumerate(pos)])
    
    flist,_ = getfrfaces(image)
    faces = np.array([(index,p[0],p[1],p[2],p[3]) for index,p in enumerate(flist)])
    
    if doplot:
        for i in boxes:
            colorbox(image,i[1:],color=(120,60,180),text=i[0],tcolor = (120,250,100))
        for i in faces:
            colorbox(image,i[1:],color=(240,60,120),text=i[0],tcolor = (250,250,180))
        imgplot(image)
    
    return faces,boxes


def doassign(cost):
    row_assign, col_assign = linear_sum_assignment(cost)
    row_assign = row_assign.reshape((-1, 1))  # (n,) to (n, 1) reshape
    col_assign = col_assign.reshape((-1, 1))  # (n,) to (n, 1) reshape
    indices = np.concatenate((col_assign, row_assign), axis=1)
    return indices


def faceboxmatch(faces,boxes,image=None,doplot=False):
    #faces and boxes are numpy arrays
    #they contain index,tx,ty,bx,by
    # output has box_index:face_index
    if len(faces)==0 or len(boxes)==0:
        return []
    
    if doplot and image is not None:
        for i in boxes:
            colorbox(image,i[1:],color=(120,60,180),text=i[0],tcolor = (120,250,100))
        for i in faces:
            colorbox(image,i[1:],color=(240,60,120),text=i[0],tcolor = (250,250,180))
        imgplot(image)
        
    # calculate iosb
    tl = np.dstack((np.maximum(boxes[:,1].reshape(-1,1),faces[:,1].reshape(1,-1)), np.maximum(boxes[:,2].reshape(-1,1),faces[:,2].reshape(1,-1))))
    br = np.dstack((np.minimum(boxes[:,3].reshape(-1,1),faces[:,3].reshape(1,-1)), np.minimum(boxes[:,4].reshape(-1,1),faces[:,4].reshape(1,-1))))
    
    intersection = np.maximum(0,br - tl+1).prod(axis=2)
    
    area_box = np.reshape((boxes[:,3] - boxes[:,1] + 1)*(boxes[:,4] - boxes[:,2] + 1),(-1,1))
    area_face = np.reshape((faces[:,3] - faces[:,1] + 1)*(faces[:,4] - faces[:,2] + 1),(1,-1))
    
    cost = intersection/np.minimum(area_box,area_face)
    
    col = len(faces)
    
    for j in range(col):
        maxcost = np.max(cost[:,j])
        if maxcost<0.5:
            continue
        maxpos = np.where((cost[:,j] >0.75) | (cost[:,j]==maxcost))[0]
        maxheight = np.max((boxes[:,4]-boxes[:,2])[maxpos])
        
        if len(maxpos)>=1:
            for i in maxpos:
                
                propx = min(max(0,faces[j,1] - boxes[i,1]),max(0,boxes[i,3] - faces[j,3]))/(boxes[i,3]-boxes[i,1])
                propy = 0
                minp = min(0,faces[j,2] - boxes[i,2])/(faces[j,4] - faces[j,2])
                maxp = min(1,max(0,faces[j,2] - boxes[i,2])/(boxes[i,4] - boxes[i,2]))
                propy = minp - maxp + 1

                if cost[i,j]>0.75 or cost[i,j]==maxcost and propy>0.5:
                    cost[i,j] += (boxes[i,4] - boxes[i,2])/maxheight   # add box length
 
                if propx>0 and propy>0.6:  
                    cost[i,j] = cost[i,j] + propx + propy 
                    
                elif propy<0.55:
                    cost[i,j] = (cost[i,j]-0.5)*abs(propy-0.1) 

                
    #print('cost ',cost)
    cost1 = cost.copy()
    mx = np.max(cost)
    if mx == 0:
        return []
    
    cost = 1-cost/mx
    indices = doassign(cost.T)
    indices = indices[np.argsort(indices[:,0])]
    ind2 = []
    for i,j in indices:
        if cost1[i,j]>0.61:       # 
            ind2.append([i,j])

    return ind2


# calculate iou of bbox
def bb_iou(boxA, boxB):
    
	# determine the (x, y)-coordinates of the intersection rectangle
	xA = max(boxA[0], boxB[0])
	yA = max(boxA[1], boxB[1])
	xB = min(boxA[2], boxB[2])
	yB = min(boxA[3], boxB[3])
    
	# compute the area of intersection rectangle
	interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
	
    # compute the area of both the prediction and ground-truth rectangles
	boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
	boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
	
    # compute the intersection over union by taking the intersection
	iou = interArea / float(boxAArea + boxBArea - interArea)

	return iou

# calculate iou of bbox
def bb_iosb(boxA, boxB):
    
	# determine the (x, y)-coordinates of the intersection rectangle
	xA = max(boxA[0], boxB[0])
	yA = max(boxA[1], boxB[1])
	xB = min(boxA[2], boxB[2])
	yB = min(boxA[3], boxB[3])
    
	# compute the area of intersection rectangle
	interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
	
    # compute the area of both the prediction and ground-truth rectangles
	boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
	boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
	
    # compute the intersection over union by taking the intersection
	iosb = interArea / np.min((boxAArea,boxBArea))

	return iosb

def sshimg(iname):
    import paramiko
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect("10.200.0.52",
                    username="DELL", password="azest4836", allow_agent = False)
    ftp_client=client.open_sftp()
    with ftp_client.open(r'C:\Users\Dell\Documents\azestproject\p201_activestreamhc_x\uploads\img'+'\\'+iname) as f:
        img = cv2.imdecode(np.frombuffer(f.read(), np.uint8), 1)
    
    ftp_client.close()
    client.close()
    return img



def reshapeimg(image,wshape=960,hshape=576,shape=(540,960)):
    (H, W) = image.shape[:2]
    
    if shape is not None:
        H,W = shape
    
    elif W>wshape or H>hshape:
        if H/W < 0.6:
            H = int(wshape*H/W + 0.5)
            W = wshape
        else:
            W = int(hshape*W/H + 0.5)
            H = hshape
            
    else:
        H = (int(H/32)+1)*32
        W = (int(W/32)+1)*32

    return cv2.resize(image,(W,H),interpolation = cv2.INTER_AREA)

def showvid(base,link=None,record=False,delay=0):
    #link = 'rtsp://admin:admin123@192.168.99.140:554'
    #savelink = base+'\\tracking_app\\chess_1.jpg',frame
    if link==None:
        cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    else:
        cap = cv2.VideoCapture(link)
        
    if record:
        fourcc = cv2.VideoWriter_fourcc(*'DIVX')
        out = cv2.VideoWriter(base+'/out.avi',  fourcc, 25, (960,540))
    
    imglist = []
    if delay>=0:
        a = time.time()
        while(True):
            ret, frame = cap.read()
            frame = reshapeimg(frame)
            imglist.append(frame)
            if time.time() - a > delay:
                break
            
    try: 
        while(True):
            ret, frame = cap.read()
            frame = reshapeimg(frame)
            imglist.remove(imglist[0])
            imglist.append(frame)
            cv2.imshow('frame',imglist[0])

            if record:
                out.write(frame)
            if cv2.waitKey(20) & 0xFF == ord('q'):
                #cv2.imwrite(savelink,frame)
                cap.release()
                cv2.destroyAllWindows()
                if record:
                    out.release()
                return frame.shape[:2]
    
    except KeyboardInterrupt:
        #cv2.imwrite(base+'/out2.jpg',frame)
        cap.release()
        if record:
            out.release()
        cv2.destroyAllWindows()
        return frame.shape[:2]
    
# find quality of color image
def quality(img):
    #img = np.uint8(cv2.cvtColor(img,cv2.COLOR_BGR2GRAY))
    ddepth = cv2.CV_8U
    laplacian = cv2.Laplacian(img, ddepth, ksize=3) 
    #disp(laplacian)
    return laplacian.var()

# enhance the quality of cut image using CLAHE method
def enhanceQ(img,q,cl=0.75):
    
    clahe = cv2.createCLAHE(clipLimit=cl, tileGridSize=(8,8))
    img = clahe.apply(img)
    #img[:,:,1] = clahe.apply(img[:,:,1])
    #img[:,:,2] = clahe.apply(img[:,:,2])
    
    return img

def flip(img):
    
    # np.flip(img,len(img.shape)-2)
    return cv2.flip(img, 1)

# add the annotated output and input image for better visual
def fusion(img,img2):
    img3 = cv2.add(7*np.uint8(img/12),5*np.uint8(img2/12))
    img2[img2<10] = 0
    img3[img2==0] = img[img2==0]
    return img3

def detect_color(img,bgrLower,bgrUpper):
    img_mask = cv2.inRange(img, bgrLower, bgrUpper) # BGRからマスクを作成
    result = cv2.bitwise_and(img, img, mask=img_mask)
    return result

def calcdist(p1,p2):
    return np.sqrt(np.square(p1[0]-p2[0]) + np.square(p1[1]-p2[1]))

def ldist(h):
    n,d = h.shape
    maxd = 0  # largest distance
    maxp = None
    xL,yL = mid = np.int32(np.average(h,axis=0))
    dp = None   
    mcd = 0   # maximum distance through the center
    for i in range(n):
        x1,y1 = h[i]
        xL,yL = mid
        x,y = (2*xL-x1),(2*yL-y1)
        cnt = 0

        while True:
            pos = cv2.pointPolygonTest(h,(x,y),True)
            cnt += 1
            #print(pos,x,y,xL,yL,x1,y1,' ',0)
            if abs(pos)<3 or cnt>10:
                #print(pos,x,y,xL,yL,x1,y1,' ',1)
                break

            elif pos<0 and abs(pos)>=3:
                x,y = np.int32(((x+xL)/2,(y+yL)/2))
                #print(pos,x,y,xL,yL,x1,y1,' ',2)

            elif pos>=0 and abs(pos)>=3:
                cx,cy = x,y
                x,y = (2*x-xL),(2*y-yL)
                xL,yL = cx,cy

                #print(pos,x,y,xL,yL,x1,y1,' ',3)

        cd = calcdist(h[i],[x,y])
        if mcd < cd:
            mcd = cd
            dp = [h[i],np.int32([x,y])]

        for j in range(i+1,n):
            d = calcdist(h[i],h[j])
            if maxd < d:
                maxd = d
                maxp = [h[i],h[j]]
        #print('')
    return maxd,maxp,mcd,dp





