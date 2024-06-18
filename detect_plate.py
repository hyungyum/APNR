import argparse
import time
import os
import cv2
import torch
import copy
import numpy as np
from models.experimental import attempt_load
from utils.datasets import letterbox
from utils.cv_puttext import cv2ImgAddText
from utils.general import check_img_size, non_max_suppression_face, scale_coords, non_max_suppression
from utils.torch_utils import select_device
from plate_recognition.plate_rec import get_plate_result, init_model, cv_imread
from plate_recognition.double_plate_split_merge import get_split_merge

# 색상 배열, 각 색상은 BGR 형식
clors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (0, 255, 255)]
# 위험 단어 리스트
danger = ['danger', 'warning']

# 네 점을 정렬하여 좌상단, 우상단, 우하단, 좌하단 순서로 반환하는 함수
def order_points(pts):
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]  # 좌상단
    rect[2] = pts[np.argmax(s)]  # 우하단
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]  # 우상단
    rect[3] = pts[np.argmax(diff)]  # 좌하단
    return rect

# 네 점을 이용하여 원근 변환을 수행하는 함수
def four_point_transform(image, pts):
    rect = pts.astype('float32')
    (tl, tr, br, bl) = rect
    # 상단의 두 점과 하단의 두 점 간의 거리 계산
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))
    # 좌측의 두 점과 우측의 두 점 간의 거리 계산
    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))
    # 변환 행렬 생성
    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]], dtype="float32")
    M = cv2.getPerspectiveTransform(rect, dst)
    # 원근 변환 적용
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
    return warped

# 모델을 로드하는 함수
def load_model(weights, device):
    model = attempt_load(weights, map_location=device)  # 모델 로드
    return model

# 원본 이미지의 좌표로 변환하는 함수
def scale_coords_landmarks(img1_shape, coords, img0_shape, ratio_pad=None):
    if ratio_pad is None:
        gain = min(img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1])  # 이미지 축소 비율
        pad = (img1_shape[1] - img0_shape[1] * gain) / 2, (img1_shape[0] - img0_shape[0] * gain) / 2  # 패딩 크기 계산
    else:
        gain = ratio_pad[0][0]
        pad = ratio_pad[1]

    coords[:, [0, 2, 4, 6]] -= pad[0]  # x 좌표에서 패딩 제거
    coords[:, [1, 3, 5, 7]] -= pad[1]  # y 좌표에서 패딩 제거
    coords[:, :8] /= gain  # 축소 비율 적용
    coords[:, 0].clamp_(0, img0_shape[1])  # x1
    coords[:, 1].clamp_(0, img0_shape[0])  # y1
    coords[:, 2].clamp_(0, img0_shape[1])  # x2
    coords[:, 3].clamp_(0, img0_shape[0])  # y2
    coords[:, 4].clamp_(0, img0_shape[1])  # x3
    coords[:, 5].clamp_(0, img0_shape[0])  # y3
    coords[:, 6].clamp_(0, img0_shape[1])  # x4
    coords[:, 7].clamp_(0, img0_shape[0])  # y4
    return coords

# 번호판 영역을 인식하고 변환하는 함수
def get_plate_rec_landmark(img, xyxy, conf, landmarks, class_num, device, plate_rec_model, is_color=False):
    h, w, c = img.shape
    result_dict = {}
    tl = 1 or round(0.002 * (h + w) / 2) + 1  # 선 두께

    x1 = int(xyxy[0])
    y1 = int(xyxy[1])
    x2 = int(xyxy[2])
    y2 = int(xyxy[3])
    height = y2 - y1
    landmarks_np = np.zeros((4, 2))
    rect = [x1, y1, x2, y2]
    for i in range(4):
        point_x = int(landmarks[2 * i])
        point_y = int(landmarks[2 * i + 1])
        landmarks_np[i] = np.array([point_x, point_y])

    class_label = int(class_num)
    roi_img = four_point_transform(img, landmarks_np)  # 투시 변환
    if class_label:
        roi_img = get_split_merge(roi_img)
    if not is_color:
        plate_number, rec_prob = get_plate_result(roi_img, device, plate_rec_model, is_color=is_color)  # 번호판 인식
    else:
        plate_number, rec_prob, plate_color, color_conf = get_plate_result(roi_img, device, plate_rec_model, is_color=is_color)
    result_dict['rect'] = rect  # 번호판 영역
    result_dict['detect_conf'] = conf  # 검출 신뢰도
    result_dict['landmarks'] = landmarks_np.tolist()  # 번호판 코너 좌표
    result_dict['plate_no'] = plate_number  # 번호판 번호
    result_dict['rec_conf'] = rec_prob  # 각 문자에 대한 인식 확률
    result_dict['roi_height'] = roi_img.shape[0]  # 번호판 높이
    result_dict['plate_color'] = ""
    if is_color:
        result_dict['plate_color'] = plate_color  # 번호판 색상
        result_dict['color_conf'] = color_conf  # 색상 신뢰도
    result_dict['plate_type'] = class_label  # 0: 단층 번호판, 1: 이중 번호판

    return result_dict

# 번호판 검출 및 인식을 수행하는 함수
def detect_Recognition_plate(model, orgimg, device, plate_rec_model, img_size, is_color=False):
    conf_thres = 0.3  # 신뢰도 임계값
    iou_thres = 0.5  # IoU 임계값
    dict_list = []
    img0 = copy.deepcopy(orgimg)
    assert orgimg is not None, 'Image Not Found '  # 이미지 확인
    h0, w0 = orgimg.shape[:2]
    r = img_size / max(h0, w0)
    if r != 1:
        interp = cv2.INTER_AREA if r < 1 else cv2.INTER_LINEAR  # 보간법 선택
        img0 = cv2.resize(img0, (int(w0 * r), int(h0 * r)), interpolation=interp)

    imgsz = check_img_size(img_size, s=model.stride.max())  # 이미지 크기 확인

    img = letterbox(img0, new_shape=imgsz)[0]
    img = img[:, :, ::-1].transpose(2, 0, 1).copy()  # BGR에서 RGB로 변환 및 전치

    t0 = time.time()

    img = torch.from_numpy(img).to(device)
    img = img.float()
    img /= 255.0  # 0-255에서 0.0-1.0으로 정규화
    if img.ndimension() == 3:
        img = img.unsqueeze(0)

    pred = model(img)[0]

    pred = non_max_suppression_face(pred, conf_thres, iou_thres)  # 비최대 억제 수행

    for i, det in enumerate(pred):
        if len(det):
            det[:, :4] = scale_coords(img.shape[2:], det[:, :4], orgimg.shape).round()

            for c in det[:, -1].unique():
                n = (det[:, -1] == c).sum()

            det[:, 5:13] = scale_coords_landmarks(img.shape[2:], det[:, 5:13], orgimg.shape).round()

            for j in range(det.size()[0]):
                xyxy = det[j, :4].view(-1).tolist()
                conf = det[j, 4].cpu().numpy()
                landmarks = det[j, 5:13].view(-1).tolist()
                class_num = det[j, 13].cpu().numpy()
                result_dict = get_plate_rec_landmark(orgimg, xyxy, conf, landmarks, class_num, device, plate_rec_model, is_color=is_color)
                dict_list.append(result_dict)
    return dict_list

# 결과 이미지를 그리는 함수
def draw_result(orgimg, dict_list, crime_plate_number, is_color=False):
    result_str = ""
    for result in dict_list:
        rect_area = result['rect']
        x, y, w, h = rect_area[0], rect_area[1], rect_area[2] - rect_area[0], rect_area[3] - rect_area[1]
        padding_w = 1.5 * w  # 차량 전체를 포함하도록 가로로 더 넓게 확장
        padding_h = 6.0 * h  # 차량 전체를 포함하도록 세로로 더 넓게 확장
        car_rect_area = [
            max(0, int(x - padding_w)),
            max(0, int(y - padding_h)),
            min(orgimg.shape[1], int(rect_area[2] + padding_w)),
            min(orgimg.shape[0], int(rect_area[3] + padding_h))
        ]

        height_area = result['roi_height']
        landmarks = result['landmarks']
        result_p = result['plate_no']
        if result['plate_type'] == 0:
            result_p += " "
        else:
            result_p += " " + " double layer"
        result_str += result_p + " "

        for i in range(4):
            cv2.circle(orgimg, (int(landmarks[i][0]), int(landmarks[i][1])), 5, clors[i], -1)
        cv2.rectangle(orgimg, (rect_area[0], rect_area[1]), (rect_area[2], rect_area[3]), (0, 0, 255), 2)

        if result['plate_no'] == crime_plate_number:
            cv2.putText(orgimg, "Crime Vehicle!", (car_rect_area[0], car_rect_area[1] - 20), cv2.FONT_HERSHEY_SIMPLEX,
                        2, (0, 0, 255), 3)
            cv2.rectangle(orgimg, (car_rect_area[0], car_rect_area[1]), (car_rect_area[2], car_rect_area[3]),
                          (0, 0, 255), 3)

        labelSize = cv2.getTextSize(result_p, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)  # 텍스트 크기 키움
        if rect_area[0] + labelSize[0][0] > orgimg.shape[1]:
            rect_area[0] = int(orgimg.shape[1] - labelSize[0][0])
        orgimg = cv2.rectangle(orgimg, (rect_area[0], int(rect_area[1] - round(1.6 * labelSize[0][1]))),
                               (int(rect_area[0] + round(1.2 * labelSize[0][0])), rect_area[1] + labelSize[1]),
                               (255, 255, 255), cv2.FILLED)
        if len(result) >= 1:
            orgimg = cv2ImgAddText(orgimg, result_p, rect_area[0], int(rect_area[1] - round(1.6 * labelSize[0][1])),
                                   (0, 0, 0), 30)  # 텍스트 크기 키움
    print(result_str)
    return orgimg


# 비디오 프레임에서 초당 프레임 수, 총 프레임 수 및 총 시간을 계산하는 함수
def get_second(capture):
    if capture.isOpened():
        rate = capture.get(5)
        FrameNumber = capture.get(7)
        duration = FrameNumber / rate
        return int(rate), int(FrameNumber), int(duration)
    else:
        return 0, 0, 0

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--detect_model', nargs='+', type=str, default='weights/plate_detect.pt', help='model.pt path(s)')
    parser.add_argument('--rec_model', type=str, default='weights/plate_rec_color.pth', help='model.pt path(s)')
    parser.add_argument('--is_color', type=bool, default=True, help='plate color')
    parser.add_argument('--image_path', type=str, default='imgs', help='source')
    parser.add_argument('--img_size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--output', type=str, default='result', help='source')
    parser.add_argument('--video', type=str, default='', help='source')
    parser.add_argument('--crime_plate', type=str, required=True, help='crime plate number')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    opt = parser.parse_args()
    print(opt)
    save_path = opt.output
    count = 0
    if not os.path.exists(save_path):
        os.mkdir(save_path)

    detect_model = load_model(opt.detect_model, device)
    plate_rec_model = init_model(device, opt.rec_model, is_color=opt.is_color)

    total = sum(p.numel() for p in detect_model.parameters())
    total_1 = sum(p.numel() for p in plate_rec_model.parameters())
    print("detect params: %.2fM, rec params: %.2fM" % (total / 1e6, total_1 / 1e6))

    time_all = 0
    time_begin = time.time()
    if not opt.video:
        if not os.path.isfile(opt.image_path):
            file_list = []
            allFilePath(opt.image_path, file_list)
            for img_path in file_list:
                print(count, img_path, end=" ")
                time_b = time.time()
                img = cv_imread(img_path)
                if img is None:
                    continue
                if img.shape[-1] == 4:
                    img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
                dict_list = detect_Recognition_plate(detect_model, img, device, plate_rec_model, opt.img_size, is_color=opt.is_color)
                ori_img = draw_result(img, dict_list, opt.crime_plate)
                img_name = os.path.basename(img_path)
                save_img_path = os.path.join(save_path, img_name)
                time_e = time.time()
                time_gap = time_e - time_b
                if count:
                    time_all += time_gap
                cv2.imwrite(save_img_path, ori_img)
                count += 1
            print(f"sumTime time is {time.time() - time_begin} s, average pic time is {time_all / (len(file_list) - 1)}")
        else:
            print(count, opt.image_path, end=" ")
            img = cv_imread(opt.image_path)
            if img.shape[-1] == 4:
                img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
            dict_list = detect_Recognition_plate(detect_model, img, device, plate_rec_model, opt.img_size, is_color=opt.is_color)
            ori_img = draw_result(img, dict_list, opt.crime_plate)
            img_name = os.path.basename(opt.image_path)
            save_img_path = os.path.join(save_path, img_name)
            cv2.imwrite(save_img_path, ori_img)
    else:
        video_name = opt.video
        capture = cv2.VideoCapture(video_name)
        fourcc = cv2.VideoWriter_fourcc(*'MP4V')
        fps = capture.get(cv2.CAP_PROP_FPS)
        width, height = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)), int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        out = cv2.VideoWriter('result4.mp4', fourcc, fps, (width, height))
        frame_count = 0
        fps_all = 0
        rate, FrameNumber, duration = get_second(capture)
        if capture.isOpened():
            while True:
                t1 = cv2.getTickCount()
                frame_count += 1
                print(f"Frame {frame_count}", end=" ")
                ret, img = capture.read()
                if not ret:
                    break
                img0 = copy.deepcopy(img)
                dict_list = detect_Recognition_plate(detect_model, img, device, plate_rec_model, opt.img_size, is_color=opt.is_color)
                ori_img = draw_result(img, dict_list, opt.crime_plate)
                t2 = cv2.getTickCount()
                infer_time = (t2 - t1) / cv2.getTickFrequency()
                fps = 1.0 / infer_time
                fps_all += fps
                str_fps = f'fps:{fps:.4f}'
                cv2.putText(ori_img, str_fps, (20, 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                out.write(ori_img)
        else:
            print("Failed to open video")
        capture.release()
        out.release()
        cv2.destroyAllWindows()
        print(f"all frames: {frame_count}, average fps: {fps_all / frame_count} fps")
