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
※현재 경로가 odels/research/object_detection 이어야 합니다.

※https://github.com/datitran/raccoon_dataset 의 파일을 사용했습니다.

## Data를 tf.record타입으로 변환시키자.
    !python generate_tfrecord.py --csv_input=images/train_labels.csv --image_dir=images/train --output_path=train.record
    
   TEST
  
    !python generate_tfrecord.py --csv_input=images/test_labels.csv --image_dir=images/test --output_path=test.record
    
## Pretrained_model을 다운로드
https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/tf1_detection_zoo.md
![image](https://user-images.githubusercontent.com/50165842/88173593-cd808c00-cc5d-11ea-950d-58fb085625ea.png)

여기에서 각자 원하는 모델을 다운 받으시면 됩니다.

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
"/content/models/research/object_detection/training/labelmap.pbtxt" 이 경로에 labelmap.pbtxt를 넣어주시면 됩니다.

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

## WebCamera 로 적용해보기

저는 Colab에서 진행을 하였습니다.    Colab상에서 opencv로 웹캠 화면 제어가 되지 않습니다.    
       따라서 , 저는 화면을 캡처해서 사진을 찍는 방향으로 진행하였습니다.    
<pre>
<code>
    from IPython.display import display, Javascript
from google.colab.output import eval_js
from base64 import b64decode

def take_photo(filename='photo.jpg', quality=0.8):
  js = Javascript('''
    async function takePhoto(quality) {
      const div = document.createElement('div');
      const capture = document.createElement('button');
      capture.textContent = 'Capture';
      div.appendChild(capture);

      const video = document.createElement('video');
      video.style.display = 'block';
      const stream = await navigator.mediaDevices.getUserMedia({video: true});

      document.body.appendChild(div);
      div.appendChild(video);
      video.srcObject = stream;
      await video.play();

      // Resize the output to fit the video element.
      google.colab.output.setIframeHeight(document.documentElement.scrollHeight, true);

      // Wait for Capture to be clicked.
      // await new Promise((resolve) => capture.onclick = resolve);

      const canvas = document.createElement('canvas');
      canvas.width = video.videoWidth;
      canvas.height = video.videoHeight;
      canvas.getContext('2d').drawImage(video, 0, 0);
      stream.getVideoTracks()[0].stop();
      div.remove();
      return canvas.toDataURL('image/jpeg', quality);
    }
    ''')
  display(js)
  data = eval_js('takePhoto({})'.format(quality))
  binary = b64decode(data.split(',')[1])
  with open(filename, 'wb') as f:
    f.write(binary)
  return filename

   
위 코드는 캡쳐한 화면을 저장하는 코드입니다.   
<pre>
<code>
import cv2
cap = cv2.VideoCapture(0)
try:
    with detection_graph.as_default():
        with tf.Session() as sess:
                # Get handles to input and output tensors
                ops = tf.get_default_graph().get_operations()
                
                all_tensor_names = {output.name for op in ops for output in op.outputs}
                tensor_dict = {}
                
                for key in [
                  'num_detections', 'detection_boxes', 'detection_scores',
                  'detection_classes', 'detection_masks'
                ]:
                    tensor_name = key + ':0'
                    if tensor_name in all_tensor_names:
                        tensor_dict[key] = tf.get_default_graph().get_tensor_by_name(
                      tensor_name)
               
                while True:
                    
                    ret, image_np = cap.read()
                    filename = take_photo()
                    print('Saved to {}'.format(filename))
  
                    image_open = Image.open('photo.jpg')
                    image_np = asarray(image_open)
                    image_np=image_np.copy()
                    print('ok')

                    # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
                    image_np_expanded = np.expand_dims(image_np, axis=0)
                   
                    # Actual detection.
                    output_dict = run_inference_for_single_image(image_np, detection_graph)
                    boxes = output_dict['detection_boxes']
                    print(output_dict['detection_scores'])
                    # 참조코드
                    # get all boxes from an array
                    max_boxes_to_draw = boxes.shape[0]
                    # get scores to get a threshold
                    scores = output_dict['detection_scores']
                    # this is set as a default but feel free to adjust it to your needs
                    min_score_thresh=.5
                    # iterate over all objects found
                    print(max_boxes_to_draw)
                    max_num=-1
                    for i in range(min(max_boxes_to_draw, boxes.shape[0])):
                        # 
                        if scores is None or scores[i] > min_score_thresh:
                            # boxes[i] is the box which will be drawn
                            # class_name = category_index[output_dict['detection_classes'][i]]['name']
                            box = tuple(boxes[i].tolist())
                            
                            print ("This box is gonna get used", box, output_dict['detection_classes'][i])

                    # # 참조코드
                    # Visualization of the results of a detection.
                    vis_util.visualize_boxes_and_labels_on_image_array(
                        image_np,
                        output_dict['detection_boxes'],
                        output_dict['detection_classes'],
                        output_dict['detection_scores'],
                        category_index,
                        instance_masks=output_dict.get('detection_masks'),
                        use_normalized_coordinates=True,
                        line_thickness=8)
                    
                    display(Image.fromarray(image_np))
                    # cv2_imshow('object_detection', cv2.resize(image_np, (800, 600)))
                    # if cv2.waitKey(25) & 0xFF == ord('q'):
                    #     cap.release()
                    #     cv2.destroyAllWindows()
                    #     break
                
except Exception as e:
    print(e)
    cap.release()


</code>
</pre>   
위 코드는 캡처 사진을 detection한후 저장 해주는 코드입니다.
   ![image](https://user-images.githubusercontent.com/50165842/88271139-b69b7180-cd11-11ea-9010-c72a7cb25ee1.png)    
      ![image](https://user-images.githubusercontent.com/50165842/88271179-c3b86080-cd11-11ea-8264-47aec63cc842.png)
    
    
![image](https://user-images.githubusercontent.com/50165842/88271230-d16de600-cd11-11ea-8722-c42b3999db8d.png)
</code>
</pre>  



**자세한 참조사항  https://woongjun-warehouse.tistory.com/category/%EB%94%A5%EB%9F%AC%EB%8B%9D%ED%94%84%EB%A1%9C%EC%A0%9D%ED%8A%B8/face_identification
여기서 참조하시면 됩니다.**


※계속 업로드중입니다.
