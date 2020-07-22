# Object-Detection

Tensorflow ObejectDetection API를 사용해본 것입니다.



## DataSet을 만든다



DateSet은 각자 알아서 구하시면 됩니다.
저는 제가 직접 구한 face data를 사용하였습니다.


## 이미지 라벨링 하기
https://github.com/tzutalin/labelImg

labelimg라는 프로그램을 제공해주는 깃헙입니다.

https://tzutalin.github.io/labelImg/

위는 다운로드 링크입니다.

## Data를 TfRecord 타입으로 변환시킨다.

    !python xml_to_csv.py



## Pretrained_model을 다운로드
https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/tf1_detection_zoo.md
![image](https://user-images.githubusercontent.com/50165842/88173593-cd808c00-cc5d-11ea-950d-58fb085625ea.png)

여기에서 각자 원하는 모델을 다운 받으시면 됩니다.

※필자는 tf1버전으로 진행하였다.

## Config파일을 변경후 학습을 진행한다.
자세한 참조사항  https://woongjun-warehouse.tistory.com/category/%EB%94%A5%EB%9F%AC%EB%8B%9D%ED%94%84%EB%A1%9C%EC%A0%9D%ED%8A%B8/face_identification
여기서 참조하시면 됩니다.
