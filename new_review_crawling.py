# from _typeshed import WriteableBuffer
import time
import re

import pandas as pd

from subway_data import *
from address_crawling import *
from selenium.webdriver.support.ui import Select
import os

# input_target으로 들어온 역의 주소를 찾는 함수

def search_address(target_station, target_input_address, sort_dist_flag):
    # 입력할 search의 form과 button을 찾는다.
    input_target = target_input_address.strip() #지번
    input_station = target_station.strip() #역이름+2호선
    search_text = driver.find_element_by_css_selector('#search > div > form > input')
    search_button = driver.find_element_by_css_selector('#button_search_address > button.btn.btn-default.ico-pick')
    cancel_button = driver.find_element_by_css_selector('#button_search_address > button.btn-search-location-cancel.btn-search-location.btn.btn-default > span')
    prev_url = driver.current_url

    for i in range(4):
        if i == 0:
            # driver.execute_script('arguments[0].click();', cancel_button)
            search_text.clear()
            time.sleep(1)
            # sort_selector = driver.find_element_by_class_name("list-option-inner").find_element_by_class_name("form-control").find_element_by_css_selector('option:nth-child(5)').click()
            # sort_selector = driver.find_element_by_css_selector('#content > div > div.row.restaurant-list-info > div.list-option > div > select > option:nth-child(5)')
            # 입력 form에 input_target을 넣고 검색을 한다.
            driver.find_element_by_css_selector('#search > div > form > input').click()
            search_text.send_keys(input_target)
            driver.execute_script("arguments[0].click();", search_button)
            # search_text.send_keys(Keys.ENTER)
            time.sleep(2)

        else:
        # if i == 1:
            '''
            나온 결과에서 역으로 끝나는 주소를 filtering한다. -> ex 강남역 검색시 '~~~ 강남역' 이라는 주소를 가져온다. 강남역 뒤에는 다른 글자가 없어야 된다. 
            이때 주소는 도로명 주소를 가져온다.
            '''
            # driver.execute_script('arguments[0].click();', cancel_button)
            search_text.clear()
            time.sleep(1)
            name, num = input_station.split()
            driver.find_element_by_css_selector('#search > div > form > input').click()

            if i ==1:
                search_text.send_keys(name+num) #역이름+호선이름 입력 (붙여쓰기)
            elif i==2:
                search_text.send_keys(name+' '+num) #역이름+호선이름 입력 (띄어쓰기)
            elif i==3:
                search_text.send_keys(name) #역이름

            driver.execute_script("arguments[0].click();", search_button)
            # search_text.send_keys(Keys.ENTER)
            time.sleep(2)

        try:
            if driver.find_element_by_css_selector('#search > div > form > ul > li:nth-child(1) > a').text == "검색하신 주소를 찾을 수 없습니다.":
                if i == 3:
                    return sort_dist_flag, True
                continue
            else:
                html = driver.page_source
                soup = BeautifulSoup(html, 'html.parser')
                address_list = soup.find_all('a', class_='ng-binding ng-scope')
                address_list = [address.find('i') for address in address_list]
                re_address_list = [re.sub(r'\[도로명\]', '', target.get_text()) if target is not None else 'None' for
                                   target in
                                   address_list]
                target_address = [target for target in re_address_list if
                                  target.split()[-1] == input_station.split()[0] or target.split()[-1] ==
                                  input_station.split()[1]]

                # 도로명 주소를 가져온 후 search form을 cancel_button으로 form의 text를 없애주고 얻은 도로명 주소로 다시 검색해서 해당 주소의 주변 식당을 보여주는 페이지로 넘어간다.
                if len(target_address) != 0:
                    driver.execute_script("arguments[0].click();", cancel_button)
                    time.sleep(1)
                    search_text.clear()
                    driver.find_element_by_css_selector('#search > div > form > input').click()
                    search_text.send_keys(target_address[0])
                    time.sleep(2)
                    # search_text.send_keys(Keys.ENTER)
                    driver.execute_script("arguments[0].click();", search_button)
                    time.sleep(2)

                if prev_url == driver.current_url:
                    continue
                break
        except:
            break

    if not sort_dist_flag:
        driver.find_element_by_css_selector('#content > div > div.row.restaurant-list-info > div.list-option > div > select > option:nth-child(5)').click()
        time.sleep(5)
        # driver.get(url)
        # time.sleep(5)
        sort_dist_flag = not sort_dist_flag

    return sort_dist_flag, False

# 리뷰 크롤링하는 함수, 이 부분의 크롤링 할 대상을 수정해야 한다. <- 현재는 재영님이 사용한 코드를 사용하는 중
def review_crawling(target_station, target_address, target_category):
    count = 0
    loop = True
    review_button = driver.find_element_by_css_selector('#content > div.restaurant-detail.row.ng-scope > div.col-sm-8 > ul > li:nth-child(2) > a')
    driver.execute_script("arguments[0].click();", review_button)
    time.sleep(5)

    # 리뷰 '더 보기' 클릭 ('더 보기'가 없거나 20번 누르면 스탑)
    while loop and count < 3:
        try:
            element = WebDriverWait(driver, 5).until(
                EC.presence_of_element_located((By.CSS_SELECTOR, '#review > li.list-group-item.btn-more > a'))
            )
            more_button = driver.find_element_by_css_selector('#review > li.list-group-item.btn-more > a')
            driver.execute_script("arguments[0].click();", more_button)
            count += 1
            time.sleep(1.5)
        except TimeoutException:
            loop = False

    # 페이지 소스 출력
    html = driver.page_source
    html_source = BeautifulSoup(html, 'html.parser')

    # 브랜드
    brand = html_source.find("span", attrs={"class": "restaurant-name ng-binding",
                                            "ng-bind": "restaurant.name"}).text.strip()

    # 최소주문금액
    min_cost = driver.find_element_by_css_selector("#content > div.restaurant-detail.row.ng-scope > div.col-sm-8 > div.restaurant-info > div.restaurant-content > ul > li:nth-child(3) > span").text
    
    # 여기서 얻은 html소스로 리뷰, 메뉴, 별점
    restaurant_review = html_source.find_all("li", attrs={"class": "list-group-item star-point ng-scope",
                                                          "ng-repeat": "review in restaurant.reviews"})

    for i in restaurant_review:
        user_id = i.find("span", attrs={"class": "review-id ng-binding",
                                        "ng-show": "review.phone"}).text.strip()
        time_info = i.find("span", attrs={"class": "review-time ng-binding",
                                          "ng-bind": "review.time|since"}).text.strip()
        review = i.find("p", attrs={"class": "ng-binding",
                                    "ng-bind-html": "review.comment|strip_html",
                                    "ng-show": "review.comment"}).text.strip()
        menu = i.find("div", attrs={"class": "order-items default ng-binding",
                                    "ng-click": "show_review_menu($event)"}).text.strip()
        star = len(i.find_all("span", attrs={"class": "full ng-scope",
                                             "ng-repeat": "i in review.rating|number_to_array track by $index"}))
        taste_star = i.find("span", attrs={"class": "points ng-binding",
                                           "ng-show": "review.rating_taste > 0"}).text.strip()
        quantity_star = i.find("span", attrs={"class": "points ng-binding",
                                              "ng-show": "review.rating_quantity > 0"}).text.strip()
        delivery_star = '-1' if i.find("span", attrs={"class": "points ng-binding",
                                              "ng-show": "review.rating_delivery > 0"}) == None else i.find("span", attrs={"class": "points ng-binding",
                                              "ng-show": "review.rating_delivery > 0"}).text.strip()
        
        cur_time = time.time()
        if "시간" in time_info:
            hour = re.findall('\d+', time_info)
            written_review = cur_time - float(86400 * int(hour[0]))
            written_review = time.localtime(written_review)
        elif "일주일 전" in time_info:
            written_review = cur_time - float(86400 * 7)
            written_review = time.localtime(written_review)
        elif "일 전" in time_info:
            day = re.findall('\d+', time_info)
            written_review = cur_time - float(86400 * int(day[0]))
            written_review = time.localtime(written_review)
        elif "어제" == time_info:
            written_review = cur_time - 86400
            written_review = time.localtime(written_review)
        else:
            written_review = time_info

        written_review = f"{written_review[0]}-{written_review[1]}-{written_review[2]}"

        try:
            image= i.find("table", attrs={
                                  "ng-if": "review.review_images.length == 1",
                                  "style": "width: 575px;"})
            image_urls = image.find_all("img")
            image_list = []
            for image_url in image_urls:
                img_url = image_url.select('img')[0]['ng-src']
                image_list.append(img_url)
        except:
            image_list = ['-1']
            
        print(f'호선 : {target_station}')
        print(f'주소 : {target_address}')
        print(f"분류: {target_category}")
        print(f"브랜드명: {brand}")
        print(f"유저: {user_id}")
        print(f"시간: {written_review}")
        print(f"리뷰: {review}")
        print(f"메뉴: {menu}")
        print(f"평점: {star}")
        print(f"맛 평점: {taste_star}")
        print(f"양 평점: {quantity_star}")
        print(f"배달 평점: {delivery_star}")
        print(f"이미지 url {image_list})")
        print("\n")
        main_list.append([target_station, target_address, target_category, brand, user_id, written_review, review, menu, star, taste_star, quantity_star, delivery_star])

def click_category(target):
    index = category_name.index(target) + 3
    print(target, index)
    category_button = driver.find_element_by_css_selector('#category > ul > li:nth-child('+str(index)+')')
    driver.execute_script("arguments[0].click();", category_button)
    time.sleep(5)

# 식당을 클릭하고 review_crawling 함수를 통해서 해당 식당을 크롤링한다.
def click_restaurant(target_station, target_address, target_category):
    # 현재 페이지에서 요기요 등록점 식당에 대해서 정보를 얻어온다. test할때는 한 번에 50개 정도의 식당정보가 나왔습니다.
    restaurant_list = driver.find_element_by_css_selector('#content > div > div:nth-child(5) > div > div').text.split('\n\n\n')
    number_of_restaurant = len(restaurant_list)
    print(number_of_restaurant)

    # 뒤로 가기를 하기 위해서 url정보를 저장
    prev_url = driver.current_url

    # 위에서 얻어온 식당 정보를 바탕으로 첫번째 식당부터 하나씩 클릭해서 페이지에 접근하기 + 접근한 식당 페이지에서 크롤링하기
    for i in range(1, 2):#number_of_restaurant+1):
        target_restaurant = driver.find_element_by_css_selector('#content > div > div:nth-child(5) > div > div > div:nth-child('+str(i)+') > div' )
        target_restaurant.click()
        time.sleep(2)
        review_crawling(target_station, target_address, target_category)
        time.sleep(3)
        driver.get(prev_url)
        time.sleep(3)

def address_page(target_station, target_address, sort_dist_flag, skip_flag):
    cancel_button = driver.find_element_by_css_selector('#button_search_address > button.btn-search-location-cancel.btn-search-location.btn.btn-default > span')

    sort_dist_flag, skip_flag = search_address(target_station, target_address, sort_dist_flag)

    # if not skip_flag:
    #     for target_category in category_name:
    #         click_category(target_category)
    #         click_restaurant(target_station, target_address, target_category)
    #         driver.execute_script("arguments[0].click();", cancel_button)

    return sort_dist_flag, skip_flag

# 크롤링을 정보를 담기 위한 main_list
main_list = []
url = 'https://www.yogiyo.co.kr/mobile/#/%EC%84%9C%EC%9A%B8%ED%8A%B9%EB%B3%84%EC%8B%9C/135081/'
category_name = ['1인분 주문']#, '프랜차이즈', '치킨', '피자/양식', '중국집', '한식', '일식/돈까스', '족발/보쌈', '야식', '분식', '카페/디저트']
driver = webdriver.Chrome('./chromedriver')
driver.get(url)
response = requests.get(url)

if response.status_code == 200:
    # 거리 기준 정렬을 했는지 표시하는 boolean 변수
    sort_dist_flag = False
    skip_flag = False

    # 지하철 역 정보가 있는 subway_data.csv가 없으면 생성
    if not os.path.exists('pilot_subway_address_info.csv'):
        address_crawling()

    # 지하철 호선 번호와 해당 호선의 지하철 역 정보를 가져오기
    subway_data = pd.read_csv('pilot_subway_address_info.csv', encoding='utf-8')
    # subway_data.rename(columns={'Unnamed: 0' : 'station_name'},inplace=True)
    subway_list_dict = subway_data.to_dict(orient='row')
    subway_number2 = subway_data['subway_name'][0]
    target_statation = subway_data['station_name'].to_list()
    target_station_address = subway_data['address'].to_list()
    # 지하철 역을 주소로 주면서 search address 반보고하기
    # for station, address in zip(target_statation, target_station_address):
    for station, address in zip(['신도림','역삼','강남'], ['신도림동 460-26', '역삼동 804','역삼동 858']):
        sort_dist_flag, skip_flag = address_page(station+'역 '+subway_number2, address, sort_dist_flag, skip_flag)
        skip_flag = False

    df = pd.DataFrame(main_list)
    df.to_csv('pilot.csv', encoding='utf-8')

else :
    print(response.status_code)
