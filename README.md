# Object-Detection

Tensorflow ObejectDetection API를 사용해본 것입니다.
Python 환경이 따로 없으시면 Colab에서 하시는걸 추천합니다.   
Colab에는 웬만한 라이브러리가 다운되어 있습니다.

## 환경 구비하기
<pre>
<code>

pip install --upgrade tensorflow==1.14.0
</pre>
</code>   

코랩에서 Tensorflow 버전이 오류가 나므로 버전을 낮춰줍니다.   
<pre>
<code>
!pip install Cython
!apt-get install protobuf-compiler python-pil 
</pre>
</code>   
Cython, protobuf 을 다운 받습니다.

<pre>
<code>

!git clone https://github.com/tensorflow/models
</pre>
</code>

training 에필요한 model repository를 다운받습니다.



<pre>
<code>
%%bash
cd models/research/
protoc object_detection/protos/*.proto --python_out=.


</pre>
</code>
<pre>
<code>
%set_env PYTHONPATH=/content/models/research:/content/models/research/slim
</pre>
</code>


protobuf을 컴파일 하고 OD 패키지 설치

<pre>
<code>
!python3 object_detection/builders/model_builder_test.py

</pre>
</code>

다운받은 모델이 잘 작동하는지 테스트를 합니다.

![image](https://user-images.githubusercontent.com/50165842/88275656-b5ba0e00-cd18-11ea-838a-01fe936b3746.png)   

잘 작동하면 위와 같은 화면이 나옵니다.


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
메인 코드입니다.

<pre>
<code>
    def main():
    for folder in ['train', 'test']:
        image_path = os.path.join(os.getcwd(), ('images/' + folder))
        xml_df = xml_to_csv(image_path)
        xml_df.to_csv(('images/'+folder+'_labels.csv'), index=None)
        print('Successfully converted xml to csv.')
</code>
</pre> 
        
※images폴더안의   train,test라는 폴더안에 들어잇어야합니다.
※현재 경로가 models/research/object_detection 이어야 합니다.

※https://github.com/datitran/raccoon_dataset 의 파일을 사용했습니다.

## Data를 tf.record타입으로 변환시키자.
    !python generate_tfrecord.py --csv_input=images/train_labels.csv --image_dir=images/train --output_path=train.record
    
   TEST
  
    !python generate_tfrecord.py --csv_input=images/test_labels.csv --image_dir=images/test --output_path=test.record
    
## Pretrained_model을 다운로드
https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/tf1_detection_zoo.md
![image](https://user-images.githubusercontent.com/50165842/88173593-cd808c00-cc5d-11ea-950d-58fb085625ea.png)

여기에서 각자 원하는 모델을 다운 받으시면 됩니다.
<pre>
<code>


!wget http://download.tensorflow.org/models/object_detection/ssd_mobilenet_v1_coco_2018_01_28.tar.gz 
!tar -xvf ssd_mobilenet_v1_coco_2018_01_28.tar.gz
</pre>
</code>
※파일의 다운로드 경로는 models/research/object_detection 입니다.      
※위와 같이 명령어를 쳐서 해도 됩니다.    
   
※이게 어려우면 직접 다운받으신다음에 압축을 풀어주시면 됩니다.    
   
※필자는 tf1버전으로 진행하였다.    
   
※필자는 coco모델을 다운받아 사용 하였습니다.    
    
## Config파일을 변경후 학습을 진행한다.
<pre>
<code>
    item {
    id: 1
    name: 'face'
    }

</code>
</pre> 
메모장에서 labelmap.pbtxt 라는 파일명으로 위와같은 라벨맵을 만들어줍니다.
"/content/models/research/object_detection/training/labelmap.pbtxt" 이 경로에 labelmap.pbtxt , config파일 를 넣어주시면 됩니다.   

legacy 폴더에있는 train파일을 상위폴더 object_detection에 복사해서 옮겨준다.   
**※저는 face_detection만을 하엿기 때문에 face만 라벨을 지정했습니다.**

**※본인 목적에 따라 라벨 개수를지정해주시면됩니다..**


![image](https://user-images.githubusercontent.com/50165842/88268770-ed6f8880-cd0d-11ea-8796-4a03863dd632.png)


![image](https://user-images.githubusercontent.com/50165842/88268808-f9f3e100-cd0d-11ea-925b-bb7845332a7c.png)


![image](https://user-images.githubusercontent.com/50165842/88268837-06783980-cd0e-11ea-956f-2ac1e25dcedc.png)


위의 경로를 찬찬히 따라가면 config파일들이 나옵니다.

![image](https://user-images.githubusercontent.com/50165842/88269061-56ef9700-cd0e-11ea-874f-8b4071641105.png)


Google Colab에서 진행하기 때문에 경로를 Colab에 맞춰서 바꿔줍니다.

**※config파일은 자신이 선택한 모델에 맞게 선택 하시면 됩니다.**

![image](https://user-images.githubusercontent.com/50165842/88269344-d1b8b200-cd0e-11ea-9ae2-77685805a2d8.png)

**※형광펜 칠한 부분만 수정하시면 작동합니다..**   
**※추가적으로 바꾸고 싶은 파라미터는 알아서 바꾸시면 됩니다...**
## Training
<pre>
<code>
!python train.py --logtostderr --train_dir=training/ --pipeline_config_path=training/ssd_mobilenet_v1_coco.config
</code>
</pre>    
지금 캡쳐 해놓은 화면이 없지만 , step 당 loss들이 출력됩니다.


**※현재 경로를 /content/models/research/object_detection/ 로 해주셔야 합니다.**       
## Loading Model
<pre>
<code>
!python export_inference_graph.py --input_type image_tensor --pipeline_config_path training/ssd_mobilenet_v1_coco.config --trained_checkpoint_prefix training/model.ckpt-100 --output_directory inference_graph
</code>
</pre> 
## WebCamera 로 적용해보기

저는 Colab에서 진행을 하였습니다.    Colab상에서 opencv로 웹캠 화면 제어가 되지 않습니다.    
       따라서 , 저는 화면을 캡처해서 사진을 찍는 방향으로 진행하였습니다.    

   ![image](https://user-images.githubusercontent.com/50165842/88271139-b69b7180-cd11-11ea-9010-c72a7cb25ee1.png)    
      ![image](https://user-images.githubusercontent.com/50165842/88271179-c3b86080-cd11-11ea-8264-47aec63cc842.png)
    
    
![image](https://user-images.githubusercontent.com/50165842/88271230-d16de600-cd11-11ea-8722-c42b3999db8d.png)
</code>
</pre>  



**자세한 참조사항  https://woongjun-warehouse.tistory.com/category/%EB%94%A5%EB%9F%AC%EB%8B%9D%ED%94%84%EB%A1%9C%EC%A0%9D%ED%8A%B8/face_identification
여기서 참조하시면 됩니다.**


※계속 업로드중입니다.
