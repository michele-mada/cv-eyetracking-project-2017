from selenium import webdriver
from selenium.webdriver.common.keys import Keys
import time


class Scroller:

    browser = webdriver.Firefox()
    browser.get('http://thoreau.eserver.org/walden1a.html')
    body = browser.find_element_by_xpath('/html/body')

    def scrolldown(self):
        print("scrolling down")
        self.body.send_keys(Keys.DOWN)

    def scrollup(self):
        print("scrolling up")
        self.body.send_keys(Keys.UP)
        
