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

![image](https://user-images.githubusercontent.com/50165842/88174852-eab65a00-cc5f-11ea-9a20-dd26c5910a50.png)

위와 같은 프로그램이 실행되서 라벨링을 하시면 됩니다.

![image](https://user-images.githubusercontent.com/50165842/88174992-26512400-cc60-11ea-807e-ec2c25d119a8.png)




라벨링을 한 이미지와 xml들을  train,test로 나누어서images 라는 폴더에 저장한후 models/research/object_detection 안에  됩니다. 

## xml를 csv 타입으로 변환시킨다.

    !python xml_to_csv.py

※images폴더안의   train,test라는 폴더안에 들어잇어야합니다.
※현재 경로가 odels/research/object_detection 이어야 합니다.

※https://github.com/datitran/raccoon_dataset 의 파일을 사용했습니다.

## Data를 tf.record타입으로 변환시키자.
    !python generate_tfrecord.py --csv_input=images/train_labels.csv --image_dir=images/train --output_path=train.record
    
    !python generate_tfrecord.py --csv_input=images/test_labels.csv --image_dir=images/test --output_path=test.record
    
## Pretrained_model을 다운로드
https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/tf1_detection_zoo.md
![image](https://user-images.githubusercontent.com/50165842/88173593-cd808c00-cc5d-11ea-950d-58fb085625ea.png)

여기에서 각자 원하는 모델을 다운 받으시면 됩니다.

※필자는 tf1버전으로 진행하였다.

## Config파일을 변경후 학습을 진행한다.
자세한 참조사항  https://woongjun-warehouse.tistory.com/category/%EB%94%A5%EB%9F%AC%EB%8B%9D%ED%94%84%EB%A1%9C%EC%A0%9D%ED%8A%B8/face_identification
여기서 참조하시면 됩니다.
