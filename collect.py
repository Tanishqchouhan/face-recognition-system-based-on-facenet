import cv2import oscam = cv2.VideoCapture(0)cam.set(3, 640) cam.set(4, 480) face_detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml' )face_id = str(input('enter Employee name :'))print("face capture. camera m dekho bhiya")count = 0os.makedirs('dataset/Employee/' + face_id, exist_ok = True)a = 'dataset/Employee/' + face_id + '/'while(True):    ret, img = cam.read()        faces = face_detector.detectMultiScale(img, 1.03, 4)    for (x,y,w,h) in faces:        cv2.rectangle(img, (x,y), (x+w,y+h), (255,0,0), 2)             count += 1               cv2.imwrite( a + str(face_id) + '.' + str(count) + ".jpg", img[y:y+h,x:x+w])        cv2.imshow('image', img)    k = cv2.waitKey(100) & 0xff    if k == 27:        break    elif count >= 50:         breakprint("Exiting Program and cleanup stuff")cam.release()cv2.destroyAllWindows()