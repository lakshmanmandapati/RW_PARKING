from ultralytics import YOLO

# Load a pre-trained YOLOv12-M model for high accuracy
model = YOLO('yolov12m.pt')

# Train the model
# The `data` argument points to your data.yaml file.
results = model.train(data='data.yaml', 
                      epochs=150, 
                      imgsz=1280, 
                      batch=16,
                      name='yolov12_pklot_high_accuracy')

# The `epochs` parameter is set higher to ensure the model has enough time to learn complex features for accuracy.
# The `imgsz` (image size) is increased to 1280, which can improve accuracy for detecting smaller objects.
# The `batch` size is set based on typical GPU memory for optimal training speed.
# The `name` parameter gives a name to your training run's results folder.