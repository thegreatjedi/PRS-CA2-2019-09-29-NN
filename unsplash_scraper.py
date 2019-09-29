from math import inf
from pathlib import Path
from pickle import dump, load
from urllib.request import urlretrieve

from selenium.webdriver import Chrome
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support.expected_conditions \
    import visibility_of_all_elements_located
from selenium.webdriver.support.ui import WebDriverWait


def scrape_images(search_term: str, num_to_scrape: int = 0, min_size: int = 0)\
        -> None:
    browser = Chrome(executable_path="./chromedriver.exe")
    pickle_file_name = f"./{search_term}_scraped_images.pickle"
    file_path = Path(pickle_file_name)
    if file_path.is_file():
        file = open(pickle_file_name, mode="rb")
        scraped_file_ids = load(file)
        file.close()
    else:
        scraped_file_ids = set()
    file = open(pickle_file_name, mode="wb")
    
    try:
        browser.get(f"https://unsplash.com/s/photos/{search_term}")
        Path(f"./images/{search_term}").mkdir(parents=True, exist_ok=True)
        html_elem = browser.find_element_by_tag_name("html")
        new_links = set()
        portrait_links = set()
        scrape_more = True
        
        while scrape_more:
            img_elems = WebDriverWait(browser, inf).until(
                visibility_of_all_elements_located(
                    (By.XPATH, "//div[@class='IEpfq']/img[@src]")))
            for elem in img_elems:
                link = elem.get_attribute("src")
                url_segments = link.split("?")
                file_id = url_segments[0].split("/")[-1].split("photo-")[-1]
                args = url_segments[1].split("&")
                img_id = [arg for arg in args if "ixid=" in arg]
                if len(img_id) > 0 and file_id not in scraped_file_ids:
                    new_links.add(link)
                    img_width = elem.size["width"]
                    img_height = elem.size["height"]
                    if img_height > img_width:
                        portrait_links.add(link)
            
            num_new_links = len(new_links)
            if len(new_links) < num_to_scrape:
                html_elem.send_keys(Keys.END)
                print(f"Links scraped to date: {num_new_links}", end="\r")
            else:
                scrape_more = False
                print(f"Total links to download: {num_new_links}")
        
        i = 0
        for link in new_links:
            print(f"Parsing link {i}: {link}", end="\r")
            url_segments = link.split("?")
            file_id = url_segments[0].split("/")[-1].split("photo-")[-1]
            get_args = url_segments[1].split("&")
            img_id = [arg for arg in get_args if "ixid=" in arg]
            img_id = img_id[0]

            img_url = f"{url_segments[0]}?{img_id}&fm=png&fit=max"
            if link in portrait_links:
                img_url += f"&w={min_size}"
            else:
                img_url += f"&h={min_size}"
            urlretrieve(img_url, f"./images/{search_term}/{file_id}.png")
            scraped_file_ids.add(file_id)
            i += 1
        print("All links parsed.")
    finally:
        browser.quit()
        dump(scraped_file_ids, file)
        file.close()


'''
if __name__ == "__main__":
    scrape_images(search_term="car", num_to_scrape=5000, min_size=256)
    scrape_images(search_term="bus", num_to_scrape=5000, min_size=256)
    scrape_images(search_term="truck", num_to_scrape=5000, min_size=256)
    scrape_images(search_term="motorcycle", num_to_scrape=5000, min_size=256)
'''
