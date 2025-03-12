from ultralytics import YOLO

def get_roll(image):
    # Load a model
    model = YOLO(r"E:\OCR PROJECT - new version\models\roll_no.pt")
    # Predict with the model
    results = model(image)[0]
    c=[]
    for detection in results.boxes.data.tolist():
        
        x1, y1, x2, y2, score, class_id = detection

        # Create output directory for the specific class (if it doesn't exist)
        class_name = model.names[int(class_id)]
        c.append(class_name)
    roll_no=''
    for i in c:
        roll_no=roll_no+i
        
    return roll_no
