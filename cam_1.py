# import the library
import cv2
import numpy as np
import tensorflow as tf
  
# define a video capture object
vid = cv2.VideoCapture(0)

#check for the camera
if not (vid.isOpened()):
    print("Could not open video device")
    ok = False
else:
    ok = True


### load model and print summary
lmodel = tf.keras.models.load_model("CNN_deeper_bin.h5") 
lmodel.summary()


num_classes = 2
class_names = ['Acceptable', 'Nonacceptable']

#Image cropped
left=280
right=750
top=350
bottom=590

#image testing size
#base image : 480x640
left=85
right=555
top=120
bottom=360


while(ok):  
    # Capture the video frame
    # by frame
    ret, frame = vid.read()
    if ret == False:
        print("Can not get frame.")
        break

    #frame transforming
    frame = frame[top:bottom, left:right]
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    size = 180, 100

    # Display the resulting frame
    cv2.imshow('frame', frame)

    frame = np.resize(frame, size)
    frame = tf.convert_to_tensor(frame)
    frame = tf.expand_dims(frame, 0) # Create a batch


    #Prediction and result part
    predictions = lmodel.predict(frame)
    score = predictions#tf.nn.softmax(predictions[0])
    
    print("SCORE : \n",score)
    print("\nPredicitons :\n",predictions)
    
    
    print(
        "This image most likely belongs to {} with a {:.2f} percent confidence. "
        .format(class_names[np.argmax(score)], 100 * np.max(score))
    )


    # the 'q' button is set as the
    # quitting button you may use any
    # desired button of your choice
    if cv2.waitKey(0) & 0xFF == ord('q'):
        break
  
# After the loop release the cap object
vid.release()
# Destroy all the windows
cv2.destroyAllWindows()