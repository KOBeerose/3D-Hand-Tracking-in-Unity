import cv2
import numpy as np

# Object class to insert logo
class Object:
    def __init__(self, start_x=100, start_y=100, size=50):
        self.logo_org = cv2.imread('Orange.png')
        self.size = size
        self.logo = cv2.resize(self.logo_org, (size, size))
        img2gray = cv2.cvtColor(self.logo, cv2.COLOR_BGR2GRAY)
        _, logo_mask = cv2.threshold(img2gray, 1, 255, cv2.THRESH_BINARY)
        self.logo_mask = logo_mask
        self.x = start_x
        self.y = start_y
        self.on_mask = False
    def insert_object(self, frame):
        roi = frame[self.y:self.y + self.size, self.x:self.x + self.size]
        roi[np.where(self.logo_mask)] = 0
        roi += self.logo
    def update_position(self, mask):
        height, width = mask.shape
        # Check if object is overlapping with moving parts
        roi = mask[self.y:self.y + self.size, self.x:self.x + self.size]
        check = np.any(roi[np.where(self.logo_mask)])
        # If object has moving parts, then find new position
        if check:
            # To save the best possible movement
            best_delta_x = 0
            best_delta_y = 0
            best_fit = np.inf
            # Try 8 different positions
            for _ in range(8):
                # Pick a random position
                delta_x = np.random.randint(-15, 15)
                delta_y = np.random.randint(-15, 15)
                # Ensure we are inside the frame, if outside, skip and continue
                if self.y + self.size + delta_y > height or self.y + delta_y < 0 or \
                        self.x + self.size + delta_x > width or self.x + delta_x < 0:
                    continue
                # Calculate how much overlap
                roi = mask[self.y + delta_y:self.y + delta_y + self.size, self.x + delta_x:self.x + delta_x + self.size]
                check = np.count_nonzero(roi[np.where(self.logo_mask)])
                # If perfect fit (no overlap), just return
                if check == 0:
                    self.x += delta_x
                    self.y += delta_y
                    return
                # If a better fit found, save it
                elif check < best_fit:
                    best_fit = check
                    best_delta_x = delta_x
                    best_delta_y = delta_y
            # After for-loop, update to best fit (if any found)
            if best_fit < np.inf:
                self.x += best_delta_x
                self.y += best_delta_y
                return

# Get the webcam (default webcam is 0)
cap = cv2.VideoCapture(0)
# If your webcam does not support 640 x 480, this will find another resolution
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
# To detect movement (to get the background)
background_subtractor = cv2.createBackgroundSubtractorMOG2()
# This will create an object
obj = Object()
# Loop forever (or until break)
while True:
    # Read the a frame from webcam
    _, frame = cap.read()
    # Flip the frame
    frame = cv2.flip(frame, 1)
    # Get the foreground mask (it is gray scale)
    fg_mask = background_subtractor.apply(frame)
    # Convert the gray scale to black and white with a threshold
    # Change the 250 threshold fitting your webcam and needs
    # - Setting it lower will make it more sensitive (also to noise)
    _, fg_mask = cv2.threshold(fg_mask, 250, 255, cv2.THRESH_BINARY)
    # Find a new position for object (logo)
    # - fg_mask contains all moving parts
    # - updated position will be the one with least moving parts
    obj.update_position(fg_mask)
    # Insert the object into the frame
    obj.insert_object(frame)
    # Show the frame in a window
    cv2.imshow('WebCam', frame)
    # To see the fg_mask uncomment the line below
    # cv2.imshow('fg_mask', fg_mask)
    # Check if q has been pressed to quit
    if cv2.waitKey(1) == ord('q'):
        break
# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()