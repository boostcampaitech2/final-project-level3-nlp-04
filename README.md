# 리뷰, 직접 쓰지 말고 이제는 FooReview로!
## 1. Team

### 😎 팀원 소개
   <div align="center">
      
   |<img src="https://user-images.githubusercontent.com/87477828/134484215-53286763-0836-4fb5-b64b-eed926890003.png" height=200 width=200>|<img src="https://avatars.githubusercontent.com/u/46557183?v=4" height=200 width=200>|<img src="https://user-images.githubusercontent.com/49185035/134527286-6fa2bcfb-ee28-47b7-bf33-0d67c2a92093.jpg" height=200 width=200>|
   |:---:|:---:|:---:|
   |[김신곤](https://github.com/SinGonKim)|[김재영](https://github.com/kimziont)|[박세진](https://github.com/pseeej)|
      
   
   |<img src="https://user-images.githubusercontent.com/52475378/134624873-d0345cf3-d0b6-48b9-a1c2-45aa47f5f677.JPG" height=200 width=200>|<img src="https://user-images.githubusercontent.com/22788924/134502594-83db95a2-c9db-46a1-9e63-8ec176f8fb89.jpeg" height=200 width=200>|<img src="https://user-images.githubusercontent.com/45033215/134625788-fdf023fd-3fc4-47d7-8f30-8dedbcfc2877.png" height=200 width=200>|<img src="https://user-images.githubusercontent.com/45033215/134476503-0e05f1cd-6e37-4a84-9701-ad9616888f3e.png" height=200 width=200>|
   |:---:|:---:|:---:|:---:|
   |[손희락](https://github.com/raki-1203)|[심우창](https://github.com/whatchang)|[이상준](https://github.com/sangjun-Leee)|[전상민](https://github.com/sangmandu)|
  
   </div>
   
### 😎 팀원별 역할
- `전상민` **PM**, Review Generation, Star Classification
- `김신곤` Data Crawling, Retrieval
- `김재영` Data Crawling, Preprocessing, Image Generation
- `박세진` Data Preprocessing, Chatbot
- `손희락` Retrieval, Image Generation, Recommendation
- `심우창` Data Crawling, Text Style Transfer
- `이상준` Review Generation, Star Classification, Text Style Transfer

## 2. FooReview
### 2.1 Data
![image](https://user-images.githubusercontent.com/49185035/147207482-2fe3af95-d7ca-4d45-bad7-48fc1b5e6343.png)
서울의 2호선 지하철역 51개를 배달 장소로 선정하여 **_메뉴, 별점, 리뷰, 카테고리 등_** 의 주요 정보를 **_요기요_** 에서 크롤링 진행하여 AWS의 MySQL DB에 저장하도록 하였습니다.
### 2.2 Review Generation
### 2.2.1 Review Re-Tagging 
EDA 진행 중, 별점과 리뷰가 상이한 경우를 확인하였습니다. 이에 따라, **유저가 준 별점은 신뢰도가 떨어지는 지표**라고 판단했고, 리뷰에 대한 새로운 별점 지표를 마련하고자 했습니다. 별점(1~5점)별로 샘플링한 약 14000개의 데이터를 바탕으로 **_요리 관련 별점, 배달 관련 별점을 1, 3, 5 점으로 태깅_** 을 진행하였습니다.  
![image](https://user-images.githubusercontent.com/49185035/147208764-37d5e440-5b5e-4015-b1db-ef2d01de876c.png)

### 2.2.2 Classification Model
Re-Tagging한 데이터들을 바탕으로 학습을 진행하였을 때, 각 모델은 f1 score에서 다음과 같은 차이를 보였습니다. 이에 따라, 별점 Re-Tagging을 위한 모델로는 **_RoBERTa-Large로 5-fold를 돌린 모델_** 을 사용하였습니다.
![image](https://user-images.githubusercontent.com/49185035/147212520-f6e0b853-5b89-49f4-b62b-39690a97a96f.png)    
해당 모델로 크롤링한 60만개의 데이터의 점수를 다시 정의하였습니다. 이 데이터들은 **_Huggingface Dataset_** 으로 관리되었고, 이는 **_리뷰 생성과 키워드 검색의 학습 데이터로 사용_** 되었습니다.
![image](https://user-images.githubusercontent.com/49185035/147212656-e1767903-1553-440f-a019-f6bc0a351906.png)

