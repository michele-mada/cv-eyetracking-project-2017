from . import selenium_based, interface


app_controllers = {
    "firefox": selenium_based.SeleniumFirefox,
    "chrome": selenium_based.SeleniumChrome,
    "None": interface.AppControlInterface,  # dummy no-op interface
}