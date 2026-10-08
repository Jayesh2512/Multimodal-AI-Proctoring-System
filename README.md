# 🎓 AI-Based Smart Online Examination & Proctoring System

An advanced, intelligent web-based examination platform engineered with automated real-time AI proctoring, facial biometrics, head-pose estimation, gaze tracking, and window event telemetry. Built as an end-to-end Computer Science academic minor project.

---

## 📌 Table of Contents
- [Project Overview](#-project-overview)
- [Key Features](#-key-features)
- [AI Proctoring Pipeline](#-ai-proctoring-pipeline)
- [System Architecture](#-system-architecture)
- [Tech Stack](#-tech-stack)
- [Directory Structure](#-directory-structure)
- [Database Configuration](#-database-configuration)
- [Installation & Quickstart](#-installation--quickstart)
- [Walkthrough & Testing Guide](#-walkthrough--testing-guide)
- [Academic Defense & Viva Key Points](#-academic-defense--viva-key-points)

---

## 📖 Project Overview

With the global transition toward remote and hybrid education, maintaining academic integrity in digital assessments is crucial. Traditional online exams are vulnerable to unauthorized external assistance, impersonation, device misuse, and screen navigation.

**AI Smart Proctoring System** provides an automated, non-intrusive supervision suite that runs directly within standard web browsers. It eliminates the need for expensive third-party proctoring centers while guaranteeing multi-layered verification and high detection accuracy.

---

## 🚀 Key Features

### 👤 Role-Based Portals

#### 1. Professor / Examiner Portal
- **Exam Management**: Create, configure, update, and manage exams.
- **Multiple Assessment Modes**:
  - **Objective (MCQ)**: Single/multi-choice, automatic scoring, negative marking, randomized question order.
  - **Subjective (Long QA)**: Essay and descriptive questions with grading rubrics.
  - **Practical / Programming**: Multi-language code evaluation environment with inputs and execution feedback.
- **Live Monitoring Dashboard**: Real-time examination supervisor feed showing active candidate statuses.
- **Proctoring Incident Inspector**: Audit timestamped logs of gaze anomalies, head movements, ambient voice spikes, phone sightings, and window switching.
- **Result Publishing**: Publish results, compute analytics, and export student performance records.

#### 2. Student / Examinee Portal
- **Biometric Face Verification**: Webcam face capture matched against student registration profile.
- **Seamless Exam Console**: Question navigation grid, bookmarking, built-in scientific calculator, and resilient timer (persists across page reloads).
- **Exam History & Results**: Review past attempts, marks obtained, and instructor feedback.
- **Helpdesk & Issue Reporting**: In-app query submission to course administrators.

---

## 🧠 AI Proctoring Pipeline

The platform runs real-time computer vision analysis on webcam frames captured at continuous intervals:

| Proctoring Dimension | Detection Technique | Action / Flag |
| :--- | :--- | :--- |
| **Identity Verification** | OpenCV DNN / DeepFace Biometric Comparison | Prevents impersonation at login and exam entry |
| **Candidate Presence** | Multi-face detection via OpenCV DNN Net | Flags **No Person** or **Multiple Persons** in frame |
| **Object Detection** | Mobile phone classification via COCO / YOLO | Flags unauthorized smartphone or device usage |
| **Head Pose Estimation** | 3D Perspective-n-Point / Normalized facial offsets | Detects abnormal head rotation (Left, Right, Up, Down) |
| **Gaze Tracking** | Eye region luminance & pupil gradient analysis | Flags candidate looking away from screen |
| **Audio Monitoring** | Web Audio API RMS decibel sampling | Logs ambient acoustic spikes and suspicious speech |
| **Browser Integrity** | HTML5 Page Visibility API & blur event listeners | Logs tab switching, window resizing, and desktop navigation |
| **Anti-Cheating Security** | JavaScript clipboard & keybinding blockers | Disables Cut, Copy, Paste, PrintScreen, and Context Menu |

---

## 🏗 System Architecture

```mermaid
graph TD
    Client[Web Browser Client] -->|HTTP / REST / AJAX| Flask[Flask Web Application Server]
    Client -->|Webcam Video Frames| CamEndpoint[/video_feed Endpoint]
    Client -->|Audio & Window Events| EventEndpoint[/window_event Endpoint]

    subgraph AI Proctoring Engine
        CamEndpoint --> FaceDet[OpenCV DNN Face Detector]
        FaceDet --> PersonCount{Person Count Check}
        PersonCount -->|0 Faces| FlagNoPerson[Flag: No Person]
        PersonCount -->|1 Face| NormalCandidate[Candidate Verified]
        PersonCount -->|>1 Faces| FlagMultiPerson[Flag: Multiple Persons]

        NormalCandidate --> HeadPose[Head Pose Estimator]
        NormalCandidate --> GazeTrack[Gaze & Pupil Tracker]
        NormalCandidate --> ObjDetect[Phone & Object Detector]
    end

    subgraph Data Layer
        Flask --> DBAdapter[DB Adapter Auto-Switch]
        DBAdapter -->|Default Fallback| SQLite[(SQLite Database: quizapp.db)]
        DBAdapter -->|Optional Production| MySQL[(MySQL Database: quizapp)]
    end

    AI Proctoring Engine -->|Store Proctoring Logs| DBAdapter
    EventEndpoint -->|Store Window Telemetry| DBAdapter
```

---

## 🛠 Tech Stack

- **Backend Framework**: Python 3, Flask, Werkzeug
- **Database**: SQLite3 (automatic zero-config) / MySQL (production option)
- **Computer Vision & AI**: OpenCV (`cv2`), OpenCV DNN, DeepFace, NumPy
- **Frontend**: HTML5, Vanilla CSS, Bootstrap, JavaScript, SweetAlert2, LottieFiles
- **Form Handling & Validation**: Flask-WTF, WTForms, WTForms-Components
- **Session & Security**: Flask-Session, CORS, Base64 Image Processing

---

## 📁 Directory Structure

```text
MyProctor.ai/
│
├── app.py                      # Main Flask application & routing controller
├── camera.py                   # AI computer vision & proctoring processing module
├── db_adapter.py               # Resilient DB wrapper (SQLite fallback + MySQL support)
├── face_detector.py            # OpenCV DNN face detection routines
├── face_landmarks.py           # Facial landmark detection module
├── objective.py                # MCQ & objective test management
├── subjective.py               # Subjective exam grading & handling
├── requirements.txt            # Modern Python package dependencies
│
├── DB/
│   ├── quizapp.db              # SQLite database (pre-initialized, zero-config)
│   └── quizappstructure.sql    # Complete SQL database schema
│
├── models/
│   ├── opencv_face_detector_uint8.pb   # OpenCV DNN quantized face detection weights
│   ├── opencv_face_detector.pbtxt      # Model configuration graph
│   └── classes.TXT                     # COCO object detection class labels
│
├── gaze_tracking/              # Eye & pupil tracking package
│   ├── __init__.py
│   ├── gaze_tracking.py
│   ├── eye.py
│   ├── pupil.py
│   └── calibration.py
│
├── static/                     # CSS stylesheets, UI assets, and client JavaScript
│   ├── css/                    # Pixel UI & Bootstrap stylesheets
│   ├── questions/              # Sample exam CSV templates
│   └── ...
│
└── templates/                  # Jinja2 HTML templates
    ├── index.html              # Modern landing page
    ├── layout.html             # Master layout template
    ├── login.html              # Secure biometric login page
    ├── register.html           # Student & professor registration
    ├── student_dashboard.html  # Student management portal
    ├── professor_dashboard.html# Examiner management portal
    ├── give_test.html          # Exam console with live AI proctoring HUD
    └── ...
```

---

## 💾 Database Configuration

The project features a **smart dual-mode database adapter** (`db_adapter.py`):
- **Default Mode (SQLite)**: Uses `DB/quizapp.db` out of the box with zero external configuration. No database installation or service configuration is required.
- **Production Mode (MySQL)**: If `Flask-MySQLdb` and MySQL are configured, the adapter automatically connects to MySQL.

---

## ⚙ Installation & Quickstart

### Prerequisites
- Python 3.9+ (Python 3.10, 3.11, 3.12, or 3.13)
- Modern web browser (Chrome, Edge, or Firefox)
- Working webcam (for facial verification & proctoring)

### 1. Clone or Open the Workspace
```powershell
cd c:\Users\ACER\Desktop\MyProctor.ai-AI-BASED-SMART-ONLINE-EXAMINATION-PROCTORING-SYSYTEM-main
```

### 2. (Optional) Create a Virtual Environment
```powershell
python -m venv venv
venv\Scripts\activate
```

### 3. Install Dependencies
```powershell
pip install -r requirements.txt
```

### 4. Run the Application
```powershell
python app.py
```

### 5. Access the Web Application
Open your web browser and navigate to:
```
http://127.0.0.1:5000
```

> **Note on Email & OTP Testing**:
> If SMTP credentials are not configured, the system's safe mail handler automatically prints all OTP verification codes and notification emails directly to the server terminal console so you can copy and verify without hindrance.

---

## 🧪 Walkthrough & Testing Guide

### Step 1: Register an Account
1. Open `http://127.0.0.1:5000/register`.
2. Select your account type: **Professor** or **Student**.
3. Allow webcam access and click **Capture Image** for biometric enrollment.
4. Fill in Name, Email, and Password, then submit.
5. Check your server terminal console for the **5-digit verification OTP** and enter it on the verification page.

### Step 2: Login with Biometric Verification
1. Open `http://127.0.0.1:5000/login`.
2. Enter your registered email, password, and select your role.
3. Align your face with the webcam to capture the verification frame.
4. The system validates credentials and facial biometric features, granting entry to the dashboard.

### Step 3: Create an Exam (Professor)
1. Go to **Professor Dashboard** -> **Create Exam**.
2. Select exam mode: **Objective (MCQ)**, **Subjective**, or **Practical**.
3. Enter duration, schedule, and passing marks.
4. Upload questions manually or via CSV template (`static/questions/testt.csv`).

### Step 4: Take the Exam with Live AI Proctoring (Student)
1. From **Student Dashboard**, click **Begin Exam**.
2. Face verification confirms candidate identity before unlocking questions.
3. While taking the exam:
   - Live AI HUD analyzes presence, head pose, and eye movement.
   - Any attempt to switch tabs or open other programs is recorded in the proctoring audit log.
4. Submit the exam to view confirmation statistics.

### Step 5: Review Proctoring Audit Logs (Professor)
1. In **Professor Dashboard**, navigate to **Live Monitoring** or **Student Logs**.
2. Inspect the timestamped breakdown of candidate images, noise levels, gaze telemetry, and window changes.

---

## 🎯 Academic Defense & Viva Key Points

When presenting this minor project to faculty examiners:
1. **Architecture Novelty**: Explain how client-side frame acquisition pairs with backend OpenCV DNN processing to achieve real-time proctoring with minimal network bandwidth.
2. **Resilience & Fault Tolerance**: Mention that the system automatically handles missing hardware models, SMTP offline states, and database fallbacks gracefully without crashing.
3. **Multi-Modal Cheating Detection**: Highlight that proctoring is not just face detection—it simultaneously analyzes **biometrics, head angle (yaw/pitch), eye gaze, ambient acoustics, and OS window blur events**.
4. **Data Privacy**: Facial data is processed into numeric matrices/features and stored in secure database logs rather than exposed publicly.

---

## 📄 License
This project is prepared for educational and academic evaluation purposes.
