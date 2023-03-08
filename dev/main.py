import numpy as np
import socket
import codecs
import imutils

import os, sys, glob
import shutil

import binascii, json, base64

from io import BytesIO
import PIL.Image
from IPython.display import Image

import cv2
import pytesseract

# from yolo_module import yolo_m


def Changing(photo):
    thresh = 205
    img_blur = cv2.GaussianBlur(photo, (5, 5), 0)
    image_BlN = cv2.medianBlur(img_blur, 7)
    return image_BlN
def blur(photo):
    return cv2.GaussianBlur(photo, (1, 1), 7)
def array_to_image(a, fmt='jpeg'):
    f = BytesIO()    
    PIL.Image.fromarray(a).save(f, fmt)    
    return IPython.display.Image(data=f.getvalue())
def kontrast_usual (photo):
    thresh = 128
    return cv2.threshold(photo, thresh, 255, cv2.THRESH_BINARY)[1]
def GettingImagetoBuff(photo):
    img_bytes = BytesIO() # getting image without buffer
    img_bytes.write(data)
    nparr = np.frombuffer(img_bytes.getvalue(), np.uint8) #buffer to array
    return cv2.imdecode(nparr, cv2.IMREAD_UNCHANGED)

if __name__ == '__main__':
    # yolo_instance = yolo_m.YOLO(weights='yolo_module/data/best.pt')

    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

    # ---- get hostnames and IP
    local_hostname = socket.gethostname()
    local_fqdn = socket.getfqdn() 
    ip_adress = socket.gethostbyname(local_hostname)

    # ---- output hostname, domain name and IP address
    print ("working on %s (%s) with %s" % (local_hostname, local_fqdn, ip_adress))
    server_adress2 = (ip_adress, 23458)
    sock.bind(server_adress2)
    print ('starting up on %s:%s' % server_adress2)
    sock.listen(1)
        
    try:
        while True:
            print ('waiting for a connection')
            connection, client_adress = sock.accept()
            try:
                print ('connection from', client_adress)
                success = 0
                getdata = connection.recv(16384*16)
                print(getdata)
                getdata=getdata.decode()
                result = json.loads(getdata)
                print(result["data"])
                image = BytesIO(base64.b64decode(result["data"]))
                id = result["id"]
                pilimage = PIL.Image.open(image)
                pilimage.save("server_results/res.jpg")
                text = None
                success = 0
                if os.path.exists("server_results/res.jpg"):
                    os.system('python yolov5/detect4.py --weights yolov5/runs/train/yolov5s_results6/weights/best.pt --conf-thres 0.4 --source server_results/res.jpg')
                if os.path.exists("server_results/exp/res_0.jpg"):
                    for imageName in glob.glob('server_results/exp/*.jpg'): #assuming JPG
                        print("\n")
                        name='server_results/exp/*.jpg'
                        image = cv2.imread("server_results/exp/res_0.jpg") 
                        image = blur(image)
                text = pytesseract.image_to_string(image, lang = "rus") #text recognition
                print("text is "+text)
                newtext=text.strip()
                s=len(newtext)
                if s>0:
                    success = 1
                    text = text.strip()
                else:
                    text = None
                response = {"id":id, "data": text, "success":success } #answer in json
                response_json = json.dumps(response)
                print(response)
                connection.send(response_json.encode()) #sending result
            except Exception as e1:
                if not getdata:
                    print("no data got through connection")
                exception_type, exception_object, exception_traceback = sys.exc_info()
                filename = exception_traceback.tb_frame.f_code.co_filename
                line_number = exception_traceback.tb_lineno
                print("Exception type: ", exception_type)
                print("File name: ", filename)
                print("Line number: ", line_number)
            finally:
                print('finish')
                #----------- Clean up the connection
                if os.path.exists("server_results/res.jpg"):
                    os.remove("server_results/res.jpg")
                if os.path.exists("server_results/exp"):
                    shutil.rmtree("server_results/exp")
                connection.close()
    except KeyboardInterrupt:    
        print('keyboard interrupted')
    except Exception as e:
        print(e)    
    finally:
        print('Goodbye')
        sock.close()
