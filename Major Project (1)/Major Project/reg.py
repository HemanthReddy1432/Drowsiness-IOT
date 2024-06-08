import cv2
import os
import tkinter as tk
from tkinter import simpledialog
from tkinter import messagebox
import face_recognition

path = 'Training_images'

if not os.path.exists(path):
    os.makedirs(path)

def register_person(name):
    cap = cv2.VideoCapture(0)
    while True:
        success, img = cap.read()
        if not success:
            messagebox.showerror("Error", "Could not read from the camera.")
            break
        cv2.imshow("Registration", img)
        key = cv2.waitKey(1)
        if key == ord('s'):
            return img, os.path.join(path, f"{name}.jpg")
        elif key == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()
    return None, None 

def is_person_registered(known_encodings, current_encoding):
    # Compare the current face encoding with known encodings
    matches = face_recognition.compare_faces(known_encodings, current_encoding)
    return any(matches)

def register_with_gui():
    name = simpledialog.askstring("Input", "Enter the name of the person to register:")
    
    if name:
        # Check if the person has already been registered
        known_encodings = []
        for file in os.listdir(path):
            image_path = os.path.join(path, file)
            img = face_recognition.load_image_file(image_path)
            encoding = face_recognition.face_encodings(img)
            if encoding:
                known_encodings.append(encoding[0])

        img, image_path = register_person(name)
        if img is not None:
            current_encoding = face_recognition.face_encodings(img)
            if current_encoding and not is_person_registered(known_encodings, current_encoding[0]):
                cv2.imwrite(image_path, img)
                messagebox.showinfo("Success", f"{name} registered successfully!\nImage saved as: {image_path}")
            else:
                os.remove(image_path) if os.path.exists(image_path) else None  # Remove the image if it exists
                messagebox.showinfo("Info", f"{name} has already been registered.")
        else:
            messagebox.showinfo("Info", "Registration canceled.")

if __name__ == "__main__":
    root = tk.Tk()
    root.withdraw()  

    register_with_gui()
    root.quit()
