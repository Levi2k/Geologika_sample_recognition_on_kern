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

import yolo_module
from utils.custom_logger import CustomLogger

if __name__ == '__main__':    

    logger = CustomLogger(name='glk-slr')

    yolo_instance = yolo_module.YOLO(weights='data/best.pt')

    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)

    # ---- get hostnames and IP
    local_hostname = socket.gethostname()
    local_fqdn = socket.getfqdn() 
    ip_addr = socket.gethostbyname(local_hostname)

    # ---- output hostname, domain name and IP address
    # logger.info ("working on %s (%s) with %s" % (local_hostname, local_fqdn, ip_addr))
    server_addr = (ip_addr, 8000)
    sock.bind(server_addr)

    logger.info('starting up on %s:%s' % server_addr)
    sock.listen(1)

    
    while True:
        try:
           logger.debug ('expecting connection...')
           connection, client_adress = sock.accept()
           logger.info ('%s connected', client_adress)

           data = b''
           # ---- recieve request
           with connection:
               while True:
                   recv_data = connection.recv(2**1)
                   # logger.debug(len(recv_data))
                   if recv_data:
                       data+=recv_data
                       # logger.debug('+') 

                   if len(recv_data)<2**1:
                       # logger.debug('break')
                       break
                       
               # logger.warning(data)  
               # request = json.loads( data.decode() )
               if data:
                    # logger.debug(data)  
                    request = json.loads( data.decode() )
                    img = np.array( 
                       PIL.Image.open(
                           BytesIO(base64.b64decode(request["data"]))
                           ))
                   
                    results = yolo_instance.detect(img)

                   # ---- recognised text-images to text

                    response = {
                           'id': request['id'], 
                           'data': [], 
                           'success': False 
                           } 

                    for i,result in enumerate(results):
                       image = cv2.GaussianBlur(result['img'], (1, 1), 7)
                       text = pytesseract.image_to_string(image, lang = "rus+eng").strip() #text recognition
                       
                       if len(text):
                           response['success'] = True

                       entry = {
                           'text':text,
                           'bbox':result['bbox'],
                           'conf':result['conf']
                       }

                       response['data'].append(entry)                        
                       logger.debug("text%d: %s" % (i,text))
                       

                    response_json = json.dumps(response)
                    logger.debug(response)
                    connection.send(response_json.encode())
               else:
                    logger.warning('no data recieved')
        except KeyboardInterrupt:    
           logger.warning('keyboard interrupted')
           sock.close()
           logger.warning('socket closed')
           break
        except Exception as e:
           logger.error(e) 
           raise e
           break 
