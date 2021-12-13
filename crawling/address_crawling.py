import time

import requests
from bs4 import BeautifulSoup
import pandas as pd
from selenium import webdriver
from selenium.webdriver.common.keys import Keys

def address_crawling():
    subway_df = pd.read_csv('../data/subway_data.csv', encoding='utf-8')
    subway_df.rename(columns={'Unnamed: 0' : 'subway_name'}, inplace=True)

    subway_name_list = subway_df['subway_name'].to_list()
    station_info_list = subway_df['station_info'].to_list()

    init_url = 'https://map.kakao.com/'
    driver = webdriver.Chrome('./chromedriver')
    driver.get(init_url)
    search_text = driver.find_element_by_css_selector('#search\.keyword\.query')
    # search_button = driver.find_element_by_css_selector('#container > shrinkable-layout > div > app-base > search-input-box > div > div.search_box > button.button_search')
    address_info_list = []


    subway_name_list = subway_name_list[1]

    for index, target_subway_name in enumerate([subway_name_list]):
    # for index, target_subway_name in enumerate(subway_name_list):
        index = 1
        for target_station_info in station_info_list[index].split(', '):
            search_keyword = target_station_info
            try:
                search_text.clear()
                search_text.send_keys(search_keyword+'역 ' + target_subway_name)
                search_text.send_keys(Keys.ENTER)
            except:
                search_text.clear()
                search_text.send_keys(search_keyword + '역 ' + target_subway_name + '선')
                search_text.send_keys(Keys.ENTER)
            time.sleep(2)
            html = driver.page_source
            soup = BeautifulSoup(html, 'html.parser')
            place_list = soup.find('ul', attrs={"class": "placelist"})
            first_list_block = place_list.find('li')
            info_item = first_list_block.find('div', attrs={"class": "info_item"})
            detail_address_info = info_item.find('div', attrs={"class":'addr'})
            prev_filtering_address = detail_address_info.text
            if '(지번)' in prev_filtering_address:
                detail_address = prev_filtering_address.split('(지번) ')[1]
            else:
                detail_address = prev_filtering_address

            address_info_list.append([target_subway_name, target_station_info, detail_address])

    df = pd.DataFrame(address_info_list, columns=['subway_name', 'station_name', 'address'])
    df.to_csv('../data/pilot_subway_address_info.csv', encoding='utf-8', index=False)
