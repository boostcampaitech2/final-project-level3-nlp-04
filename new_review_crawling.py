# from _typeshed import WriteableBuffer
from db.sql_helper import SqlHelper
from subway_data import *
from address_crawling import *
from datetime import datetime
import os
import config as c

from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException
from tqdm import tqdm
import pandas as pd
import time
from bs4 import BeautifulSoup
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager

import argparse

def get_option_chrome():
    chrome_options = webdriver.ChromeOptions()
    chrome_options.add_argument('--headless')
    chrome_options.add_argument('--no-sandbox')
    chrome_options.add_argument("--single-process")
    chrome_options.add_argument("--disable-dev-shm-usage")
    driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=chrome_options)
    return driver


# input_target으로 들어온 역의 주소를 찾는 함수

def search_address(driver, target_station, target_input_address, sort_dist_flag):
    # 입력할 search의 form과 button을 찾는다.
    input_target = target_input_address.strip()  # 지번
    input_station = target_station.strip()  # 역이름+2호선
    search_text = driver.find_element(By.CSS_SELECTOR, '#search > div > form > input')
    search_button = driver.find_element(By.CSS_SELECTOR, '#button_search_address > button.btn.btn-default.ico-pick')
    cancel_button = driver.find_element(By.CSS_SELECTOR, '#button_search_address > button.btn-search-location-cancel.btn-search-location.btn.btn-default > span')
    prev_url = driver.current_url
    search_keyword = input_target

    for i in range(4):
        if i == 0:
            search_text.clear()
            time.sleep(1)
            search_text.send_keys(search_keyword)
            driver.execute_script("arguments[0].click();", search_button)
            time.sleep(2)

        else:
            '''
            나온 결과에서 역으로 끝나는 주소를 filtering한다. -> ex 강남역 검색시 '~~~ 강남역' 이라는 주소를 가져온다. 강남역 뒤에는 다른 글자가 없어야 된다. 
            이때 주소는 도로명 주소를 가져온다.
            '''
            search_text.clear()
            time.sleep(1)
            name, num = input_station.split()

            if i == 1:
                search_keyword = name + num
                search_text.send_keys(search_keyword)  # 역이름+호선이름 입력 (붙여쓰기)
            elif i == 2:
                search_keyword = name + ' ' + num
                search_text.send_keys(search_keyword)  # 역이름+호선이름 입력 (띄어쓰기)
            elif i == 3:
                search_keyword = name
                search_text.send_keys(search_keyword)  # 역이름

            driver.execute_script("arguments[0].click();", search_button)
            time.sleep(2)

        try:
            if driver.find_element(By.CSS_SELECTOR, '#search > div > form > ul > li:nth-child(1) > a').text == "검색하신 주소를 찾을 수 없습니다.":
                if i == 3:
                    return driver, sort_dist_flag, True, None
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
                    search_keyword = target_address[0]
                    search_text.send_keys(search_keyword)
                    time.sleep(2)
                    driver.execute_script("arguments[0].click();", search_button)
                    time.sleep(2)

                if prev_url == driver.current_url:
                    continue
                break
        except:
            break

    if not sort_dist_flag:
        driver.find_element(By.CSS_SELECTOR, '#content > div > div.row.restaurant-list-info > div.list-option > div > select > option:nth-child(5)').click()
        time.sleep(5)
        sort_dist_flag = not sort_dist_flag

    return driver, sort_dist_flag, False, search_keyword


# 리뷰 크롤링하는 함수, 이 부분의 크롤링 할 대상을 수정해야 한다. <- 현재는 재영님이 사용한 코드를 사용하는 중
def review_crawling(driver, target_station, target_address, target_category):
    count = 0
    loop = True
    review_button = driver.find_element(By.CSS_SELECTOR, '#content > div.restaurant-detail.row.ng-scope > div.col-sm-8 > ul > li:nth-child(2) > a')
    driver.execute_script("arguments[0].click();", review_button)
    time.sleep(5)
    total_review_num = int(driver.find_element(By.CSS_SELECTOR, '#content > div.restaurant-detail.row.ng-scope > div.col-sm-8 > ul > li.active > a > span').text)
    # 리뷰 '더 보기' 클릭 ('더 보기'가 없거나 20번 누르면 스탑)

    while loop and count < 20:
        try:
            current_page_num = len(
                BeautifulSoup(driver.page_source, 'html.parser').find('ul', attrs={'id': 'review'}).find_all('li')) - 2

            if current_page_num == total_review_num:
                break

            element = WebDriverWait(driver, 5).until(
                EC.presence_of_element_located((By.CSS_SELECTOR, '#review > li.list-group-item.btn-more > a'))
            )
            more_button = driver.find_element(By.CSS_SELECTOR, '#review > li.list-group-item.btn-more > a')
            driver.execute_script("arguments[0].click();", more_button)
            WebDriverWait(driver, 1.5).until(
                EC.visibility_of_all_elements_located((By.CSS_SELECTOR, '#review > li:nth-child('+ str(current_page_num + 5) +')'))
            )
            count += 1
        except TimeoutException:
            loop = False

    # 페이지 소스 출력
    html = driver.page_source
    html_source = BeautifulSoup(html, 'html.parser')

    # 브랜드
    brand = html_source.find("span", attrs={"class": "restaurant-name ng-binding",
                                            "ng-bind": "restaurant.name"}).text.strip()

    # 최소주문금액
    min_cost = driver.find_element(By.CSS_SELECTOR, "#content > div.restaurant-detail.row.ng-scope > div.col-sm-8 > div.restaurant-info > div.restaurant-content > ul > li:nth-child(3) > span").text

    # 여기서 얻은 html소스로 리뷰, 메뉴, 별점
    restaurant_review = html_source.find_all("li", attrs={"class": "list-group-item star-point ng-scope",
                                                          "ng-repeat": "review in restaurant.reviews"})

    for review_index, i in enumerate(restaurant_review):
        try:
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

            delivery_star = '-1' if i.find("span", attrs={"class": "points ng-binding", "ng-show": "review.rating_delivery > 0"}) == None else i.find("span", attrs={"class": "points ng-binding",
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
            elif '분 전' in time_info or '초 전' in time_info:
                written_review = cur_time
                written_review = written_review.split()
            else:
                written_review = time_info
                written_review = re.sub(r'년|월|일', '', written_review)
                written_review = written_review.split()

            written_review = f"{written_review[0]}-{written_review[1]}-{written_review[2]}"

            try:
                image = i.select('#review > li:nth-child(' + str(review_index + 2) + ') > table')
                image_urls = image[0].find_all("img")
                image_list = []
                for image_url in image_urls:
                    img_url = image_url['data-url']
                    image_list.append(img_url)
            except:
                image_list = ['-1']

            image_str = ' '.join(image_list)

            print(f'sumway_number : {target_station}')
            print(f'address : {target_address}')
            print(f"category: {target_category}")
            print(f"restaurant_name: {brand}")
            print(f"min_cost: {min_cost}")
            print(f"user_id: {user_id}")
            print(f"review_create_time: {written_review}")
            print(f"review_context: {review}")
            print(f"menu: {menu}")
            print(f"total_star: {star}")
            print(f"taste_star: {taste_star}")
            print(f"quantity_star: {quantity_star}")
            print(f"delivery_star: {delivery_star}")
            print(f"image_str: {image_str}")
            print("\n")


            main_list.append([brand, target_station, target_address, user_id, written_review, review, menu,
                              star, taste_star, quantity_star, delivery_star, image_str, min_cost])
        except :
            continue

    return driver

def click_category(driver, target, search_address_keyword):
    index = category_name.index(target) + 3
    print(target, index)

    current_url = driver.current_url
    driver.quit()
    os.system('pkill chrome')
    driver = get_option_chrome()
    # driver = webdriver.Chrome('../pythonProject1/chromedriver')
    driver.get(current_url)
    try:
        WebDriverWait(driver, 10).until(
            EC.visibility_of_all_elements_located(By.CSS_SELECTOR, '#search > div > form > input'))
        search_text = driver.find_element(By.CSS_SELECTOR, '#search > div > form > input')
        search_button = driver.find_element(By.CSS_SELECTOR, '#button_search_address > button.btn.btn-default.ico-pick')
        search_text.send_keys(search_address_keyword)
        driver.execute_script("arguments[0].click();", search_button)

        WebDriverWait(driver, 5).until(
            EC.visibility_of_all_elements_located(
                (By.CSS_SELECTOR,
                 '#content > div > div:nth-child(5) > div'))
        )
    except:
        driver.refresh()
        driver.find_element(By.CSS_SELECTOR, '#content > div > div.row.restaurant-list-info > div.list-option > div > select > option:nth-child(5)').click()
    category_button = driver.find_element(By.CSS_SELECTOR, '#category > ul > li:nth-child(' + str(index) + ')')
    driver.execute_script("arguments[0].click();", category_button)
    WebDriverWait(driver, 5).until(
        EC.visibility_of_all_elements_located(
            (By.CSS_SELECTOR,
             '#content > div > div:nth-child(5) > div'))
    )

    return driver

# 식당을 클릭하고 review_crawling 함수를 통해서 해당 식당을 크롤링한다.
def click_restaurant(driver, target_station, target_address, target_category):
    # 현재 페이지에서 요기요 등록점 식당에 대해서 정보를 얻어온다. test할때는 한 번에 50개 정도의 식당정보가 나왔습니다.
    restaurant_list = driver.find_element(By.CSS_SELECTOR, '#content > div > div:nth-child(5) > div > div').text.split(
        '\n\n\n\n')
    number_of_restaurant = len(restaurant_list)
    print(number_of_restaurant)

    # 뒤로 가기를 하기 위해서 url정보를 저장
    prev_url = driver.current_url

    # 위에서 얻어온 식당 정보를 바탕으로 첫번째 식당부터 하나씩 클릭해서 페이지에 접근하기 + 접근한 식당 페이지에서 크롤링하기
    # for i in range(1, number_of_restaurant+1):
    for i in range(1, int((number_of_restaurant + 1) / 4)):
        target_restaurant_name = restaurant_list[i - 1].split()[0]
        if category_dict.get(target_restaurant_name) != None:
            category_value = category_dict[target_restaurant_name]
            if target_category not in category_value:
                category_value.append(target_category)
            continue

        category_dict[target_restaurant_name] = [target_category]
        target_restaurant = driver.find_element(By.CSS_SELECTOR, '#content > div > div:nth-child(5) > div > div > div:nth-child(' + str(i) + ') > div')
        target_restaurant.click()
        WebDriverWait(driver, 2).until(
            EC.visibility_of_all_elements_located(
                (By.CSS_SELECTOR, '#content > div.restaurant-detail.row.ng-scope > div.col-sm-8 > div.restaurant-info > div.restaurant-title > span'))
        )
        driver = review_crawling(driver, target_station, target_address, target_category)
        driver.get(prev_url)
        start = time.time()
        WebDriverWait(driver, 3).until(
            EC.visibility_of_all_elements_located(
                (By.CSS_SELECTOR,
                 '#content > div > div:nth-child(5) > div'))
        )
        end = time.time()
        print(end-start)

    return driver


def address_page(driver, target_station, target_address, sort_dist_flag, skip_flag):
    # cancel_button = driver.find_element(By.CSS_SELECTOR, '#button_search_address > button.btn-search-location-cancel.btn-search-location.btn.btn-default > span')

    driver, sort_dist_flag, skip_flag, search_address_keyword = search_address(driver, target_station, target_address, sort_dist_flag)

    if not skip_flag:
        for target_category in category_name:
            driver = click_category(driver, target_category, search_address_keyword)
            driver = click_restaurant(driver, target_station, search_address_keyword, target_category)
            # driver.execute_script("arguments[0].click();", cancel_button)

    return driver, sort_dist_flag, skip_flag


if __name__ == '__main__':
    os.system('pkill chrome')

    parser = argparse.ArgumentParser()
    parser.add_argument('--num', required=True, type=int, help='자신이 맡은 번호를 입력해주세요.')
    args = parser.parse_args()

    # 서버에서 실행 시 수행
    driver = get_option_chrome()

    # driver = webdriver.Chrome('../pythonProject1/chromedriver')

    # 크롤링을 정보를 담기 위한 main_list
    # main_dict = dict()
    main_list = []
    category_dict = dict()
    url = 'https://www.yogiyo.co.kr/mobile/#/%EC%84%9C%EC%9A%B8%ED%8A%B9%EB%B3%84%EC%8B%9C/135081/'
    category_name = ['1인분 주문', '프랜차이즈', '치킨', '피자/양식', '중국집', '한식', '일식/돈까스', '족발/보쌈', '야식', '분식', '카페/디저트']
    driver.get(url)
    response = requests.get(url)
    start_time = time.time()

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

        start_point = (args.num-1) * 7
        end_point = 7 * args.num
        if args.num == 7:
            end_point = len(target_statation)


        # 지하철 역을 주소로 주면서 search address 반보고하기
        for station, address in zip(target_statation[start_point:end_point], target_station_address[start_point:end_point]):
        # for station, address in zip(['홍대입구', '건대입구'], ['동교동 165', '화양동 7-3']):
            driver, sort_dist_flag, skip_flag = address_page(driver, station + '역 ' + subway_number2, address, sort_dist_flag, skip_flag)
            skip_flag = False

            df_main = pd.DataFrame(main_list)
            category_dict_new = {k: ', '.join(v) for k, v in category_dict.items()}
            df_category = pd.DataFrame.from_dict(category_dict_new, orient='index').rename(columns={0: 'category_name'})
            df_category = df_category.reset_index().rename(columns={'index': 'restaurant_name'})
            df_main.rename(columns={0: 'restaurant_name', 1: 'subway', 2: 'address', 3: 'user_id',
                                    4: 'review_create_time', 5: 'review_context', 6: 'menu',
                                    7: 'total_star', 8: 'taste_star', 9: 'quantity_star',
                                    10: 'delivery_star', 11: 'image_url', 12: 'min_cost'}, inplace=True)
            df_total = pd.merge(df_main, df_category, how='left', on='restaurant_name')

            current_time = datetime.now()

            # DB 저장 파트 config.py 에서 DB 정보 불러옴
            sql_helper = SqlHelper(host=c.HOST, port=c.PORT, db_name=c.DB_NAME, user=c.USER, passwd=c.PASSWD)

            end_crawling_time = time.time()

            sql_helper.insert(df_total)

            print(f'{station}역 {address} 배달업체 DB insert 완료!')

            # main_list 초기화
            end_work_time = time.time()

            print('*'*20)
            print(f'total crawling time : {end_crawling_time-start_time} 초')
            print('*' * 20)
            print(f'total work time : {end_work_time-start_time} 초')

            main_list = []


        # df_main.to_csv(f'pilot_main_{current_time}.csv', encoding='utf-8')
        # df_category.to_csv(f'pilot_category_{current_time}.csv', encoding='utf-8')
        # df_total.to_csv(f'pilot_total_{current_time}.csv', encoding='utf-8')

    else:
        print(response.status_code)
