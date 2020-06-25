import re, time
import pandas as pd
from urllib.parse import urlparse, urlencode, parse_qs, urlunparse

from selenium import webdriver
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import (
	ElementNotVisibleException,
	NoSuchElementException,
	StaleElementReferenceException
)

def init_driver():
	driver = webdriver.Firefox()
	driver.wait = WebDriverWait(driver, 5)
	return driver
''' End function '''


def get_image_urls_by_categories(driver, category, url):	
	images_list = []
	print("Filter images, show only category: {}".format(category))
	filter_class_element = driver.find_element_by_xpath("//div[@id='subject_list']//span[contains(text(), '{}')]".format(category))
	
	filter_class_checkbox_element = driver.find_element_by_xpath("//div[@id='subject_list']//span[contains(text(), '{}')]/preceding-sibling::span//input".format(category))
	if category in filter_class_element.text:
		print("Click on filtered category")
		WebDriverWait(driver, 30).until(
						EC.presence_of_element_located((By.XPATH, "//div[@id='subject_list']//span[contains(text(), '{}')]/preceding-sibling::span//input".format(category))))
		driver.execute_script("arguments[0].click();", filter_class_checkbox_element)
		time.sleep(5)
		
		print("Get total pages and current page number")
		page_number_element = driver.find_element_by_xpath("//input[@id='records-txt-page-num-up']")
		total_pages_element = driver.find_element_by_xpath("//input[@id='records-txt-page-num-up']//following-sibling::span[@class='spn-pages-total']")
		
		total_pages = total_pages_element.text
		num = page_number_element.get_attribute("value")
		print("values are: total: {}. current: {}".format(total_pages, num))
		
		if total_pages.isnumeric() and num.isnumeric():
			counter = 0
			page_num = int(num)
			while page_num < int(total_pages):
				if counter >= 20:
					counter = 0
					page_num += 1
					print("update to next page {}".format(page_num))
					
					next_page_element = driver.find_element_by_xpath("//button[@id='next-page-u']")
					WebDriverWait(driver, 5).until(
						EC.presence_of_element_located((By.XPATH, "//button[@id='next-page-u']")))
					driver.execute_script("arguments[0].click();", next_page_element)
					time.sleep(5)
				else:
					imgs_element = driver.find_elements_by_xpath("//div[@id='view-wrapper']//img[contains(@id, 'bookid')]")
					WebDriverWait(driver, 10).until(
						EC.presence_of_element_located((By.XPATH, "//div[@id='view-wrapper']//img[contains(@id, 'bookid')]")))
					for img_element in imgs_element:
						img_url = img_element.get_attribute("src")
						img_dict = { 'url': get_url_without_queries(img_url), 'class': category }
						images_list.append(img_dict)
						print("appended new row to images list {}".format(img_dict))

					counter += 1

		driver.execute_script("arguments[0].click();", filter_class_checkbox_element)
		time.sleep(5)
	
	image_urls = pd.DataFrame(images_list, columns=['url', 'class'])
	return image_urls
''' End function '''


def get_url_without_queries(url):
	return url.split('?', 1)[0]
''' End function '''


def main():
	driver = init_driver()
	url = "https://photos.yadvashem.org/index.html?language=en&displayType=list"
	image_categories = (
		'Women',
		'Children',
		# 'Animals',
		# 'Uniforms',
		# 'Buildings',
		# 'Street scene',
		# 'Vehicles',
		# 'Signs',
		# 'Weapons',
		# 'Railroad cars',
		# 'Nazi symbols',
		# 'Gravestones',
		# 'Barbed wire fences',
		# 'Corpses',
		# 'German soldiers',
		# 'Armband',
		# 'Snow',
		# 'Carts',
	)
	
	driver.get(url)
	time.sleep(30)
	try:
		print("Close popup if appears")
		icon_x_button = driver.find_element_by_xpath("//div[@id='ZA_CAMP_CANVAS']//img")
		driver.execute_script("arguments[0].click();", icon_x_button)
	except NoSuchElementException:
		pass
	
	for category in image_categories:
		image_urls = get_image_urls_by_categories(driver, category, url)
		file_name = './csv/{}_image_urls.csv'.format(category)
		image_urls.to_csv(file_name, encoding='utf-8', index=False)
	driver.quit()
''' End main '''


if __name__ == "__main__":
	main()