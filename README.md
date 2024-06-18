

이미지에 대한 demo 돌리는 법
```
python detect_plate.py --detect_model weights/plate_detect.pt  --rec_model weights/plate_rec_color.pth --image_path imgs --output result
```



영상에 대한 데모 돌리는 법
```
python detect_plate.py --detect_model weights/plate_detect.pt  --rec_model weights/plate_rec_color.pth --video 2.mp4
```

각 명령행 인자 <hr>
-- detect_model 검출 모델 가중치 <br>
-- rec_model 텍스트 인식 모델 가중치 <br>
-- image_path 이미지 경로 <br>
-- ouput 처리된 이미지를 출력할 경로 <br>
-- video 영상 경로 <br>


샘플 영상 다운로드 링크

[샘플 영상](https://drive.google.com/file/d/1JbwLyqpFCXmftaJY1oap8Sa6KfjoWJta/view)




