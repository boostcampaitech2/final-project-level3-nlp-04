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
전체적인 구조입니다.
<div align="center"><img src="https://user-images.githubusercontent.com/49185035/147221273-f920e176-c68b-44c3-a8a9-755c7bbfa450.png"></div>


### 2.1 Data
<div align="center"><img src="https://user-images.githubusercontent.com/49185035/147216068-c3236f58-93f4-46e0-8c7d-b9a4af94ae75.png"></div>  

서울의 2호선 지하철역 51개를 배달 장소로 선정하여 **_메뉴, 별점, 리뷰, 카테고리 등_** 의 주요 정보를 **_요기요_** 에서 크롤링 진행하여 AWS의 MySQL DB에 저장하도록 하였습니다.
### 2.2 Review Generation
### 2.2.1 Review Re-Tagging 
EDA 진행 중, 별점과 리뷰가 상이한 경우를 확인하였습니다. 이에 따라, **유저가 준 별점은 신뢰도가 떨어지는 지표**라고 판단했고, 리뷰에 대한 새로운 별점 지표를 마련하고자 했습니다. 별점(1~5점)별로 샘플링한 약 14000개의 데이터를 바탕으로 **_요리 관련 별점, 배달 관련 별점을 1, 3, 5 점으로 태깅_** 을 진행하였습니다.  
<div align="center"><img src="https://user-images.githubusercontent.com/49185035/147215250-970ff71e-253c-4f5a-9024-f9a92485981b.png"></div>  


### 2.2.2 Classification Model
Re-Tagging한 데이터들을 바탕으로 학습을 진행하였을 때, 각 모델은 f1 score에서 다음과 같은 차이를 보였습니다. 이에 따라, 별점 Re-Tagging을 위한 모델로는 **_RoBERTa-Large로 5-fold를 돌린 모델_** 을 사용하였습니다.
<div align="center"><img src="https://user-images.githubusercontent.com/49185035/147215369-38b705e6-905e-4392-8324-9b3725453d17.png"></div>  

해당 모델로 크롤링한 60만개의 데이터의 점수를 다시 정의하였습니다. 이 데이터들은 **_Huggingface Dataset_** 으로 관리되었고, 이는 **_리뷰 생성과 키워드 검색의 학습 데이터로 사용_** 되었습니다.
<div align="center"><img src="https://user-images.githubusercontent.com/49185035/147215809-63159fb9-3a7c-47ce-81bc-e804e331ceb3.png"></div>

### 2.2.3 Generation Model
리뷰 데이터에 대한 inference 결과, 모델의 크기 등을 고려하여 여러 모델들 중에서도 상대적으로 가볍고 inference 소요시간이 적게 걸리는 **_kogpt2-base-v2 모델_** 을 선택하였습니다. 

<div align="center"><img src="https://user-images.githubusercontent.com/49185035/147273576-343e1045-e4b2-4a97-a725-0af497db17b0.png"></div>


### 2.2.4 Elastic Search
생성된 리뷰와 가장 유사한 기존 리뷰를 얻고, 기존 리뷰의 이미지를 사용하는 방법으로 **_Elastic Serach를 이용하여 사진 리뷰 기능을 제공_** 하고자 하였습니다.
<div align="center"><img src="https://user-images.githubusercontent.com/49185035/147274865-975af9d3-dbed-4bbb-85cf-959b8dced6da.png"></div>

### 2.2.5 Translation standard to dialect
GRU모델은 encoder, decoder로 이용한 **_seq2seq with attention_** 방식을 이용하였습니다. 하지만 리뷰 데이터를 input으로 넣어줬을 때 생각보다 잘 바뀌지 않아 최종적으로 **_KoBART 모델_** 을 사용하였습니다.<div align="center"><img width="1035" alt="스크린샷 2021-12-22 오후 3 38 11" src="https://user-images.githubusercontent.com/22788924/147271773-c135447c-72cb-48f9-9ada-ff0b4189e72e.png"></div>


### 2.2.6 Additional Functions
- **입력받은 키워드를 기반으로 식당 추천** 후 식당 정보 제공
- 크롤링 데이터를 분석하여 **카테고리별 인기 식당 순위 제공**
- **자동화 적용**하여 매일 오전 2시 요기요 리뷰 데이터 크롤링 진행


## 3. Run FooReview
### 3.1 git clone
```python3
git clone https://github.com/boostcampaitech2/final-project-level3-nlp-04.git
cd final-project-level3-nlp-04
```

### 3.2 Requirements Install
```python3
pip install -r requirements.txt
pip install git+https://github.com/SKT-AI/KoBART#egg=kobart
```

### 3.3 Run FooReview
```python3
cd chatbot
python3 discord_bot.py
```

### 3.4 What You Need Is
- Discord Token
- DB config file

## 4. Demo Video

1. Review Generate

https://user-images.githubusercontent.com/52475378/147280764-6e4b3703-4315-44fe-ab73-d67309a3942a.MP4

2. Review based Restaurant Top3

https://user-images.githubusercontent.com/52475378/147280912-8ca96972-2a54-4e2a-bca4-6d8875224524.MP4

3. Keyword based Restaurant Top10

https://user-images.githubusercontent.com/52475378/147280994-497a35b3-cff3-4147-ba91-bb7d39315055.MP4
