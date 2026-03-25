from tools.web_fetch import fetch_url


class BrowserTool:
    """
    Sandboxed Browser Automation tool.
    Initial implementation uses web_fetch for reading pages.
    Interactive elements are placeholders for now.
    """

    def __init__(self):
        self.current_url = ""
        self.history = []

    def visit_page(self, url: str) -> str:
        """Visit a URL and return its text content."""
        self.current_url = url
        self.history.append(url)
        return fetch_url(url)

    def click_element(self, selector: str) -> str:
        """[PLACEHOLDER] Click an element on the current page."""
        return f"Browsing simulation: Clicked '{selector}' on {self.current_url}. (Interaction requires Playwright/Puppeteer driver)"

    def capture_screenshot(self, name: str = "screenshot") -> str:
        """[PLACEHOLDER] Capture a screenshot of the current page."""
        return f"Browsing simulation: Screenshot '{name}.png' captured for {self.current_url}. (Interaction requires Playwright/Puppeteer driver)"


_browser = BrowserTool()


def browser_visit(url: str) -> str:
    """Visit a website and read its content."""
    return _browser.visit_page(url)


def browser_click(selector: str) -> str:
    """Simulate a click on a web element."""
    return _browser.click_element(selector)


def browser_screenshot(name: str = "screenshot") -> str:
    """Simulate capturing a screenshot."""
    return _browser.capture_screenshot(name)
