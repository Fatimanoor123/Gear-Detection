# Gear Detection - PPE Detection using YOLO

This project detects **Personal Protective Equipment (PPE)** such as helmets, vests, and gloves in videos using the **YOLO (You Only Look Once) model**. It leverages `ultralytics` for YOLO inference and `OpenCV` for video processing.

## 📌 Features
- **Real-time PPE Detection** using a trained YOLO model.
- **Bounding Boxes & Labels** for detected safety gear.
- **Confidence Scores** to indicate detection accuracy.
- **Scalable Display** with adjustable frame resizing.

## 🛠️ Requirements
Before running the project, ensure you have the following dependencies installed:

```bash
pip install ultralytics opencv-python numpy
## 📥 Installation
Follow these steps to set up and run the project:

```bash
git clone https://github.com/your-username/GearDetection.git
cd GearDetection
pip install ultralytics opencv-python numpy
##  Project Structure
The project is structured as follows:

```bash
GearDetection/
├── best.pt                 # Pre-trained YOLO model (replace with actual model file)
├── worker.mp4              # Sample video for detection
├── detect.py               # Main script for detection
├── README.md               # Project documentation

##  🚀 How to Use
Run the following command to start detection:

```bash
python detect.py

##  🖥️ Code Overview
- Loads a **pre-trained YOLO model**.
- Reads frames from **video input**.
- Performs **detection** and draws bounding boxes.
- Displays **real-time PPE detection**.

##  🎥 Example Output
Upon running the script, the model will detect PPE and display bounding boxes over detected objects:
### Without Helmet:

![image](https://github.com/user-attachments/assets/7095cbc9-bb9d-42df-9929-fa8e39970d7c)

###  With Helmet:
![image](https://github.com/user-attachments/assets/42997ae6-96b6-43db-aa4a-19ae201e3beb)


##  🔍 Notes
- Ensure `best.pt` is correctly placed inside the project folder.
- Modify `scale = 0.3` in `detect.py` to adjust the display size.
- Replace `worker.mp4` with your own video for testing.

##  ❓ Troubleshooting
### Issue: `ModuleNotFoundError: No module named 'ultralytics'`

```bash
pip install ultralytics


## 🤝 Contributing
Contributions are welcome! If you’d like to contribute:

```bash
1. Fork the repository.
2. Create a new branch.
3. Commit your changes.
4. Submit a pull request.




##  📜 License
This project is open-source and available under the **MIT License**.


