import os
import time
import requests
import pandas as pd
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from webdriver_manager.chrome import ChromeDriverManager

# Setup Chrome options
options = webdriver.ChromeOptions()
# Uncomment the line below to run in headless mode
# options.add_argument('--headless')
options.add_argument('--no-sandbox')
options.add_argument('--disable-dev-shm-usage')
options.add_argument('--window-size=1920,1080')
options.add_argument('--ignore-certificate-errors')
options.add_argument('--disable-blink-features=AutomationControlled')
options.add_argument(
    'user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3')


# Create directories if they don't exist
def setup_directory(path):
    if not os.path.exists(path):
        os.makedirs(path)


setup_directory(r'E:\ai_real\ai_generated')
setup_directory(r'E:\ai_real\real_images')


# Function to save images with unique filenames
def save_image(url, folder_path, image_name_prefix):
    try:
        img_data = requests.get(url).content
        index = 0
        image_name = f"{image_name_prefix}_{index}.jpg"
        while os.path.exists(os.path.join(folder_path, image_name)):
            index += 1
            image_name = f"{image_name_prefix}_{index}.jpg"
        with open(os.path.join(folder_path, image_name), 'wb') as img_file:
            img_file.write(img_data)
        print(f"Saved image {image_name} from URL: {url}")
    except Exception as e:
        print(f"Could not save image {url}. Error: {e}")


# Scrape images from a website
def scrape_images(url, folder_path, class_name, num_images=100):
    driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)
    driver.set_page_load_timeout(120)  # Set page load timeout
    try:
        driver.get(url)
    except Exception as e:
        print(f"Error loading {url}: {e}")
        driver.quit()
        return []

    img_urls = set()
    scroll_pause_time = 2
    max_scroll_attempts = 10
    scroll_attempts = 0

    while len(img_urls) < num_images and scroll_attempts < max_scroll_attempts:
        # Scroll down to the bottom
        driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
        time.sleep(scroll_pause_time)

        # Get image elements
        imgs = driver.find_elements(By.TAG_NAME, "img")
        for img in imgs:
            try:
                src = img.get_attribute('src')
                if src and 'base64' not in src:  # Ignore base64 encoded images
                    img_urls.add(src)
                    if len(img_urls) >= num_images:
                        break
            except Exception as e:
                print(f"Error getting image source: {e}")

        scroll_attempts += 1

    driver.quit()

    print(f"Found {len(img_urls)} images on {url}")

    data = []
    for i, url in enumerate(img_urls):
        image_name_prefix = f"{class_name}_{i}"
        save_image(url, folder_path, image_name_prefix)
        data.append([url, class_name])

    return data


# Main function
if __name__ == "__main__":
    ai_urls = [
        "https://pixexid.com/",
        "https://stock.adobe.com/search/images?k=midjourney"
    ]

    real_urls = [
        "https://www.pexels.com/search/nature/",
        "https://www.pexels.com/search/girl/",
        "https://www.pexels.com/search/art/",
        "https://www.pexels.com/search/family%20photo/",
        "https://unsplash.com/s/photos/woman",
        "https://unsplash.com/s/photos/human",
        "https://stock.adobe.com/search?k=%22real%20flowers%22"
    ]

    all_data = []

    for ai_url in ai_urls:
        all_data.extend(scrape_images(ai_url, r'E:\ai_real\ai_generated', 'AI', num_images=5))

    for real_url in real_urls:
        all_data.extend(scrape_images(real_url, r'E:\ai_real\real_images', 'Real', num_images=5))

    # Combine data and save to CSV
    df = pd.DataFrame(all_data, columns=['URL', 'Class'])

    # Ensure the output directory exists
    output_dir = r'E:\ai_real'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    csv_path = os.path.join(output_dir, 'images_data.csv')
    if os.path.exists(csv_path):
        df_existing = pd.read_csv(csv_path)
        df = pd.concat([df_existing, df], ignore_index=True)

    df.to_csv(csv_path, index=False)
    print(f"Data saved to {csv_path}")