import cv2
import json
import os
import random
from tqdm import tqdm
import yaml

def Extract_dataset_Yolov8(json_path, split=0.8):
    json_data = json.load(open(json_path, 'r', encoding='utf-8'))

    # Tạo thư mục cho train và val
    os.makedirs("datasets/drone/images/train", exist_ok=True)
    os.makedirs("datasets/drone/labels/train", exist_ok=True)
    os.makedirs("datasets/drone/images/val", exist_ok=True)
    os.makedirs("datasets/drone/labels/val", exist_ok=True)
    os.makedirs("datasets/drone/drone.yaml", exist_ok=True)
    # Nội dung YAML
    drone_yaml = {
        "path": "./datasets/drone",
        "train": "images/train",
        "val": "images/val",
        "nc": 1,
        "names": {
            0: "object"
        }
    }

    # Ghi file
    with open("datasets/drone/drone.yaml", "w") as f:
        yaml.dump(drone_yaml, f, sort_keys=False)

    # Khởi tạo counters
    train_count = 0
    val_count = 0
    error_video = 0
    error_frame = 0

    pbar = tqdm(json_data, desc="Processing videos", ncols=100)
    
    for item in pbar:
        video_id = item['video_id']
        video_path = f"train/samples/{video_id}/drone_video.mp4"
        
        if not os.path.exists(video_path):
            error_video += 1
            pbar.set_postfix({
                "Train": train_count,
                "Val": val_count,
                "Err_vid": error_video,
                "Err_frm": error_frame
            }, refresh=True)
            continue
            
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            error_video += 1
            pbar.set_postfix({
                "Train": train_count,
                "Val": val_count,
                "Err_vid": error_video,
                "Err_frm": error_frame
            }, refresh=True)
            continue

        img_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        img_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))

        # Lấy tất cả frames có bbox của video này
        all_frames = []
        annotations = item['annotations']
        for annotation in annotations:
            bboxes = annotation['bboxes']
            for bbox in bboxes[::7]:  # Lấy mỗi frame thứ 7
                all_frames.append(bbox)
        
        # Shuffle frames của video này (với seed cố định để reproducible)
        random.seed(42)
        random.shuffle(all_frames)
        
        # Chia frames của video này thành train/val
        split_idx = int(len(all_frames) * split)
        train_frames = all_frames[:split_idx]
        val_frames = all_frames[split_idx:]
        
        # Xử lý train frames
        for bbox in train_frames:
            frame_number = bbox['frame']
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
            ret, frame = cap.read()
            
            if ret:
                x1, y1, x2, y2 = bbox['x1'], bbox['y1'], bbox['x2'], bbox['y2']
                
                box_w = x2 - x1
                box_h = y2 - y1
                x_center = (x1 + x2) / 2
                y_center = (y1 + y2) / 2
                
                x_center_norm = max(0.0, min(1.0, x_center / img_width))
                y_center_norm = max(0.0, min(1.0, y_center / img_height))
                width_norm = max(0.0, min(1.0, box_w / img_width))
                height_norm = max(0.0, min(1.0, box_h / img_height))
                
                label_path = f"datasets/drone/labels/train/{train_count}.txt"
                image_path = f"datasets/drone/images/train/{train_count}.jpg"
                
                with open(label_path, 'w') as f:
                    f.write(f"0 {x_center_norm:.6f} {y_center_norm:.6f} {width_norm:.6f} {height_norm:.6f}\n")
                
                cv2.imwrite(image_path, frame)
                train_count += 1
            else:
                error_frame += 1
        
        # Xử lý val frames
        for bbox in val_frames:
            frame_number = bbox['frame']
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
            ret, frame = cap.read()
            
            if ret:
                x1, y1, x2, y2 = bbox['x1'], bbox['y1'], bbox['x2'], bbox['y2']
                
                box_w = x2 - x1
                box_h = y2 - y1
                x_center = (x1 + x2) / 2
                y_center = (y1 + y2) / 2
                
                x_center_norm = max(0.0, min(1.0, x_center / img_width))
                y_center_norm = max(0.0, min(1.0, y_center / img_height))
                width_norm = max(0.0, min(1.0, box_w / img_width))
                height_norm = max(0.0, min(1.0, box_h / img_height))
                
                label_path = f"datasets/drone/labels/val/{val_count}.txt"
                image_path = f"datasets/drone/images/val/{val_count}.jpg"
                
                with open(label_path, 'w') as f:
                    f.write(f"0 {x_center_norm:.6f} {y_center_norm:.6f} {width_norm:.6f} {height_norm:.6f}\n")
                
                cv2.imwrite(image_path, frame)
                val_count += 1
            else:
                error_frame += 1
        
        # Cập nhật progress bar
        pbar.set_postfix({
            "Train": train_count,
            "Val": val_count,
            "Err_vid": error_video,
            "Err_frm": error_frame
        }, refresh=True)
        
        cap.release()
    
    # In kết quả tổng hợp
    print("\n" + "="*60)
    print("KẾT QUẢ TỔNG HỢP:")
    print("="*60)
    print(f"TRAIN SET:")
    print(f"  - Số mẫu: {train_count}")
    print(f"\nVAL SET:")
    print(f"  - Số mẫu: {val_count}")
    print(f"\nLỖI:")
    print(f"  - Video lỗi: {error_video}")
    print(f"  - Frame lỗi: {error_frame}")
    print(f"\nTỔNG CỘNG: {train_count + val_count} mẫu")
    print(f"TỶ LỆ TRAIN/VAL: {train_count/(train_count+val_count)*100:.1f}% / {val_count/(train_count+val_count)*100:.1f}%")
    print("="*60)

# Sử dụng
json_path = 'train/annotations/annotations.json'
Extract_dataset_Yolov8(json_path, split=0.8)