import os
import cv2
import numpy as np
import base64
import math
from face_detector import get_face_detector, find_faces

# Initialize Face Detector (quantized OpenCV DNN model provided in repo)
face_model = None
try:
    face_model = get_face_detector(quantized=True)
    print("[Camera Proctoring] OpenCV DNN Face Detector loaded successfully.")
except Exception as e:
    print(f"[Camera Proctoring Warning] Could not load quantized face detector: {e}")

# Optional YOLOv3 Model for Mobile Phone & Multiclass Detection
yolo_model = None
yolo_class_names = []
yolo_weights_path = os.path.join(os.path.dirname(__file__), 'models', 'yolov3.weights')
yolo_classes_path = os.path.join(os.path.dirname(__file__), 'models', 'classes.TXT')

if os.path.exists(yolo_weights_path):
    try:
        import tensorflow as tf
        from tensorflow.keras import Model
        from tensorflow.keras.layers import (
            Add, Concatenate, Conv2D, Input, Lambda, LeakyReLU, UpSampling2D, ZeroPadding2D, BatchNormalization
        )
        from tensorflow.keras.regularizers import l2

        def load_darknet_weights(model, weights_file):
            wf = open(weights_file, 'rb')
            major, minor, revision, seen, _ = np.fromfile(wf, dtype=np.int32, count=5)
            layers = ['yolo_darknet', 'yolo_conv_0', 'yolo_output_0', 'yolo_conv_1', 'yolo_output_1', 'yolo_conv_2', 'yolo_output_2']
            for layer_name in layers:
                sub_model = model.get_layer(layer_name)
                for i, layer in enumerate(sub_model.layers): 
                    if not layer.name.startswith('conv2d'):
                        continue
                    batch_norm = None
                    if i + 1 < len(sub_model.layers) and sub_model.layers[i + 1].name.startswith('batch_norm'):
                        batch_norm = sub_model.layers[i + 1]
                    filters = layer.filters
                    size = layer.kernel_size[0]
                    in_dim = layer.input_shape[-1]
                    if batch_norm is None:
                        conv_bias = np.fromfile(wf, dtype=np.float32, count=filters)
                    else:
                        bn_weights = np.fromfile(wf, dtype=np.float32, count=4 * filters)
                        bn_weights = bn_weights.reshape((4, filters))[[1, 0, 2, 3]]
                    conv_shape = (filters, in_dim, size, size)
                    conv_weights = np.fromfile(wf, dtype=np.float32, count=np.product(conv_shape))
                    conv_weights = conv_weights.reshape(conv_shape).transpose([2, 3, 1, 0])
                    if batch_norm is None:
                        layer.set_weights([conv_weights, conv_bias])
                    else:
                        layer.set_weights([conv_weights])
                        batch_norm.set_weights(bn_weights)
            wf.close()

        # Architecture functions
        def DarknetConv(x, filters, kernel_size, strides=1, batch_norm=True):
            if strides == 1:
                padding = 'same'
            else:
                x = ZeroPadding2D(((1, 0), (1, 0)))(x)  
                padding = 'valid'
            x = Conv2D(filters=filters, kernel_size=kernel_size, strides=strides, padding=padding,
                       use_bias=not batch_norm, kernel_regularizer=l2(0.0005))(x)
            if batch_norm:
                x = BatchNormalization()(x)
                x = LeakyReLU(alpha=0.1)(x)
            return x

        def DarknetResidual(x, filters):
            prev = x
            x = DarknetConv(x, filters // 2, 1)
            x = DarknetConv(x, filters, 3)
            return Add()([prev, x])

        def DarknetBlock(x, filters, blocks):
            x = DarknetConv(x, filters, 3, strides=2)
            for _ in range(blocks):
                x = DarknetResidual(x, filters)
            return x

        def Darknet(name=None):
            x = inputs = Input([None, None, 3])
            x = DarknetConv(x, 32, 3)
            x = DarknetBlock(x, 64, 1)
            x = DarknetBlock(x, 128, 2)  
            x = x_36 = DarknetBlock(x, 256, 8) 
            x = x_61 = DarknetBlock(x, 512, 8)
            x = DarknetBlock(x, 1024, 4)
            return Model(inputs, (x_36, x_61, x), name=name)

        def YoloConv(filters, name=None):
            def yolo_conv(x_in):
                if isinstance(x_in, tuple):
                    inputs = Input(x_in[0].shape[1:]), Input(x_in[1].shape[1:])
                    x, x_skip = inputs
                    x = DarknetConv(x, filters, 1)
                    x = UpSampling2D(2)(x)
                    x = Concatenate()([x, x_skip])
                else:
                    inputs = Input(x_in.shape[1:])
                    x = DarknetConv(inputs, filters, 1)
                x = DarknetConv(x, filters * 2, 3)
                x = DarknetConv(x, filters, 1)
                x = DarknetConv(x, filters * 2, 3)
                x = DarknetConv(x, filters, 1)
                return Model(inputs, x, name=name)(x_in)
            return yolo_conv

        def YoloOutput(filters, anchors, classes, name=None):
            def yolo_output(x_in):
                inputs = Input(x_in.shape[1:])
                x = DarknetConv(inputs, filters * 2, 3)
                x = DarknetConv(x, anchors * (classes + 5), 1, batch_norm=False)
                x = Lambda(lambda x: tf.reshape(x, (-1, tf.shape(x)[1], tf.shape(x)[2], anchors, classes + 5)))(x)
                return Model(inputs, x, name=name)(x_in)
            return yolo_output

        def yolo_boxes(pred, anchors, classes):
            grid_size = tf.shape(pred)[1]
            box_xy, box_wh, objectness, class_probs = tf.split(pred, (2, 2, 1, classes), axis=-1)
            box_xy = tf.sigmoid(box_xy)
            objectness = tf.sigmoid(objectness)
            class_probs = tf.sigmoid(class_probs)
            pred_box = tf.concat((box_xy, box_wh), axis=-1) 
            grid = tf.meshgrid(tf.range(grid_size), tf.range(grid_size))
            grid = tf.expand_dims(tf.stack(grid, axis=-1), axis=2)  
            box_xy = (box_xy + tf.cast(grid, tf.float32)) / tf.cast(grid_size, tf.float32)
            box_wh = tf.exp(box_wh) * anchors
            box_x1y1 = box_xy - box_wh / 2
            box_x2y2 = box_xy + box_wh / 2
            return tf.concat([box_x1y1, box_x2y2], axis=-1), objectness, class_probs, pred_box

        def yolo_nms(outputs, anchors, masks, classes):
            b, c, t = [], [], []
            for o in outputs:
                b.append(tf.reshape(o[0], (tf.shape(o[0])[0], -1, tf.shape(o[0])[-1])))
                c.append(tf.reshape(o[1], (tf.shape(o[1])[0], -1, tf.shape(o[1])[-1])))
                t.append(tf.reshape(o[2], (tf.shape(o[2])[0], -1, tf.shape(o[2])[-1])))
            bbox = tf.concat(b, axis=1)
            confidence = tf.concat(c, axis=1)
            class_probs = tf.concat(t, axis=1)
            scores = confidence * class_probs
            boxes, scores, classes, valid = tf.image.combined_non_max_suppression(
                boxes=tf.reshape(bbox, (tf.shape(bbox)[0], -1, 1, 4)),
                scores=tf.reshape(scores, (tf.shape(scores)[0], -1, tf.shape(scores)[-1])),
                max_output_size_per_class=100,
                max_total_size=100,
                iou_threshold=0.5,
                score_threshold=0.5
            )
            return boxes, scores, classes, valid

        def YoloV3(size=None, channels=3, anchors=None, masks=None, classes=80):
            if anchors is None:
                anchors = np.array([(10, 13), (16, 30), (33, 23), (30, 61), (62, 45),
                                    (59, 119), (116, 90), (156, 198), (373, 326)], np.float32) / 416
            if masks is None:
                masks = np.array([[6, 7, 8], [3, 4, 5], [0, 1, 2]])
            x = inputs = Input([size, size, channels], name='input')
            x_36, x_61, x = Darknet(name='yolo_darknet')(x)
            x = YoloConv(512, name='yolo_conv_0')(x)
            output_0 = YoloOutput(512, len(masks[0]), classes, name='yolo_output_0')(x)
            x = YoloConv(256, name='yolo_conv_1')((x, x_61))
            output_1 = YoloOutput(256, len(masks[1]), classes, name='yolo_output_1')(x)
            x = YoloConv(128, name='yolo_conv_2')((x, x_36))
            output_2 = YoloOutput(128, len(masks[2]), classes, name='yolo_output_2')(x)
            boxes_0 = Lambda(lambda x: yolo_boxes(x, anchors[masks[0]], classes), name='yolo_boxes_0')(output_0)
            boxes_1 = Lambda(lambda x: yolo_boxes(x, anchors[masks[1]], classes), name='yolo_boxes_1')(output_1)
            boxes_2 = Lambda(lambda x: yolo_boxes(x, anchors[masks[2]], classes), name='yolo_boxes_2')(output_2)
            outputs = Lambda(lambda x: yolo_nms(x, anchors, masks, classes), name='yolo_nms')((boxes_0[:3], boxes_1[:3], boxes_2[:3]))
            return Model(inputs, outputs, name='yolov3')

        yolo_model = YoloV3()
        load_darknet_weights(yolo_model, yolo_weights_path)
        if os.path.exists(yolo_classes_path):
            with open(yolo_classes_path, 'r') as f:
                yolo_class_names = [c.strip() for c in f.readlines()]
        print("[Camera Proctoring] YOLOv3 Model loaded successfully.")
    except Exception as yolo_err:
        print(f"[Camera Proctoring Warning] YOLOv3 loading failed ({yolo_err}). Running in OpenCV Native Proctoring mode.")
        yolo_model = None
else:
    print("[Camera Proctoring] models/yolov3.weights not found. Running in resilient OpenCV Native Proctoring mode.")


def get_frame(imgData):
    """
    Processes incoming base64 webcam frame and performs:
    1. Face detection & person count (person_status: 0=normal, 1=no face, 2=multiple faces)
    2. Mobile phone detection (mob_status: 0=none, 1=detected)
    3. Head movement estimation (user_move1: up/down, user_move2: left/right)
    4. Gaze tracking (eye_movements: 1=blinking, 2=center, 3=left, 4=right, 0=not found)
    5. HUD annotation overlay
    """
    try:
        if isinstance(imgData, str) and ',' in imgData:
            imgData = imgData.split(',', 1)[1]
        nparr = np.frombuffer(base64.b64decode(imgData), np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    except Exception as e:
        print(f"[Camera Proctoring Error] Could not decode frame: {e}")
        return {
            'jpg_as_text': imgData.encode('utf-8') if isinstance(imgData, str) else imgData,
            'mob_status': 0,
            'person_status': 0,
            'user_move1': 0,
            'user_move2': 0,
            'eye_movements': 2
        }

    if image is None:
        return {
            'jpg_as_text': b'',
            'mob_status': 0,
            'person_status': 1,
            'user_move1': 0,
            'user_move2': 0,
            'eye_movements': 0
        }

    h, w = image.shape[:2]
    mob_status = 0
    person_status = 0
    user_move1 = 0
    user_move2 = 0
    eye_movements = 2  # default center

    # 1. Face Detection & Person Count
    faces = []
    if face_model is not None:
        try:
            faces = find_faces(image, face_model)
        except Exception as e:
            print(f"[Camera Proctoring Error] Face detection error: {e}")

    num_faces = len(faces)
    if num_faces == 0:
        person_status = 1  # No person detected
    elif num_faces > 1:
        person_status = 2  # More than one person detected
    else:
        person_status = 0  # Normal (1 person)

    # 2. YOLOv3 Detection (if model loaded)
    if yolo_model is not None:
        try:
            img_yolo = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            img_yolo = cv2.resize(img_yolo, (320, 320)).astype(np.float32) / 255.0
            img_yolo = np.expand_dims(img_yolo, 0)
            boxes, scores, classes, nums = yolo_model(img_yolo)
            for i in range(nums[0]):
                cls_idx = int(classes[0][i])
                if cls_idx == 67:  # Cell phone in COCO classes
                    mob_status = 1
        except Exception as e:
            pass

    # 3. Head Pose & Gaze Analysis
    if num_faces > 0:
        # Use primary face
        primary_face = faces[0]
        fx1, fy1, fx2, fy2 = primary_face
        fw = fx2 - fx1
        fh = fy2 - fy1

        # Center of face
        fcx = fx1 + fw / 2.0
        fcy = fy1 + fh / 2.0

        # Normalized offsets from image center (-1.0 to 1.0)
        norm_x = (fcx - (w / 2.0)) / (w / 2.0)
        norm_y = (fcy - (h / 2.0)) / (h / 2.0)

        # Head Horizontal Movement (user_move2: 3=left, 4=right, 0=center)
        if norm_x > 0.30:
            user_move2 = 4  # Looking/turned Right
        elif norm_x < -0.30:
            user_move2 = 3  # Looking/turned Left
        else:
            user_move2 = 0  # Center

        # Head Vertical Movement (user_move1: 1=up, 2=down, 0=level)
        if norm_y > 0.32:
            user_move1 = 2  # Head down
        elif norm_y < -0.32:
            user_move1 = 1  # Head up
        else:
            user_move1 = 0  # Level

        # Gaze tracking: evaluate eye region in upper third of face
        try:
            ey1 = max(0, fy1 + int(fh * 0.20))
            ey2 = min(h, fy1 + int(fh * 0.55))
            ex1 = max(0, fx1 + int(fw * 0.15))
            ex2 = min(w, fx2 - int(fw * 0.15))
            eye_roi = image[ey1:ey2, ex1:ex2]

            if eye_roi.size > 0:
                eye_gray = cv2.cvtColor(eye_roi, cv2.COLOR_BGR2GRAY)
                # Left vs right brightness to deduce gaze direction
                half_w = eye_gray.shape[1] // 2
                left_half = eye_gray[:, :half_w]
                right_half = eye_gray[:, half_w:]
                left_mean = np.mean(left_half)
                right_mean = np.mean(right_half)

                # Pupil darkness imbalance can indicate lateral gaze
                diff = left_mean - right_mean
                if abs(diff) > 18:
                    eye_movements = 3 if diff > 0 else 4  # 3=left, 4=right
                else:
                    eye_movements = 2  # 2=center
        except Exception:
            eye_movements = 2

    # 4. Draw HUD Overlays on Output Frame
    annotated = image.copy()

    # Draw Face bounding boxes
    for idx, (x1, y1, x2, y2) in enumerate(faces):
        if person_status == 2:
            box_color = (0, 0, 255)  # Red for warning
            label = f"FLAGGED: Person {idx + 1}"
        else:
            box_color = (0, 230, 115)  # Vibrant Green
            label = "Candidate: Verified"

        cv2.rectangle(annotated, (x1, y1), (x2, y2), box_color, 2)
        cv2.putText(annotated, label, (x1, max(18, y1 - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, box_color, 2)

    # Draw Status HUD banner at top
    hud_bg = np.zeros((40, w, 3), dtype=np.uint8)
    status_text = "STATUS: NORMAL"
    status_color = (0, 255, 0)
    if person_status == 1:
        status_text = "WARNING: NO PERSON DETECTED"
        status_color = (0, 165, 255)
    elif person_status == 2:
        status_text = "WARNING: MULTIPLE PERSONS DETECTED"
        status_color = (0, 0, 255)
    elif mob_status == 1:
        status_text = "WARNING: MOBILE PHONE DETECTED"
        status_color = (0, 0, 255)

    gaze_labels = {0: "Unknown", 1: "Blinking", 2: "Center", 3: "Left", 4: "Right"}
    head_labels_lr = {0: "Center", 3: "Left", 4: "Right"}
    head_labels_ud = {0: "Level", 1: "Up", 2: "Down"}

    metrics_text = f"Gaze: {gaze_labels.get(eye_movements, 'Center')} | Head: {head_labels_ud.get(user_move1, 'Level')}-{head_labels_lr.get(user_move2, 'Center')}"

    cv2.putText(annotated, status_text, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, status_color, 2)
    cv2.putText(annotated, metrics_text, (w - 320, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)

    # Encode back to JPEG and Base64
    ret, jpeg = cv2.imencode('.jpg', annotated)
    jpg_as_text = base64.b64encode(jpeg)

    proctorDict = {
        'jpg_as_text': jpg_as_text,
        'mob_status': mob_status,
        'person_status': person_status,
        'user_move1': user_move1,
        'user_move2': user_move2,
        'eye_movements': eye_movements
    }

    return proctorDict