from selenium import webdriver
from selenium.webdriver.common.keys import Keys

from app_control.interface import AppControlInterface


landing_page = "http://thoreau.eserver.org/walden1a.html"


class SeleniumScroller(AppControlInterface):

    def __init__(self, browser, page=landing_page):
        super().__init__()
        self.browser = browser
        self.browser.get(page)
        self.body = self.browser.find_element_by_xpath('/html/body')

    def scroll_down(self, amount):
        print("scrolling down")
        self.body.send_keys(Keys.DOWN)

    def scroll_up(self, amount):
        print("scrolling up")
        self.body.send_keys(Keys.UP)


class SeleniumFirefox(SeleniumScroller):

    def __init__(self, page=landing_page):
        super().__init__(webdriver.Firefox(), page)


class SeleniumChrome(SeleniumScroller):
    def __init__(self, page=landing_page):
        super().__init__(webdriver.Chrome(), page)
