git clone
```
git clone https://github.com/hyungyum/APNR.git
cd APNR

```

필요한 패키지 설치
```
pip install -r requirements.txt

```


샘플 영상에 대한 데모 돌리는 법
```
python detect_plate.py --detect_model weights/plate_detect.pt  --rec_model weights/plate_rec_color.pth --video sample.mp4 --crime_plate AP05JE0
```

각 명령행 인자 <hr>
-- detect_model 검출 모델 가중치 <br>
-- rec_model 텍스트 인식 모델 가중치 <br>
-- image_path 이미지 경로 <br>
-- ouput 처리된 이미지를 출력할 경로 <br>
-- video 영상 경로 <br>
--crime_plate 범죄차량번호 <br>


샘플 영상 다운로드 링크

[샘플 영상](https://drive.google.com/file/d/1JbwLyqpFCXmftaJY1oap8Sa6KfjoWJta/view)




