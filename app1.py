import requests
import json
from bs4 import BeautifulSoup
# from outputs import output_json
import re
import unicodedata
from sqlalchemy import null

# CONFIGURATION
headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/107.0.0.0 Safari/537.36"
}
query = "dầu gội clear bạc hà 630g"
params = {
    "q": query,
    "sa": "X",
    "rlz": "1C1CHBF_enVN1029VN1029",
    "biw": 1920,
    "bih": 594,
    "tbm": "shop",
    "sxsrf": "ALiCzsbyAmZEMDMgOsuuXxBnHYzF-XNYBA:1667137473581",
    "ei": 'wX9eY9TgIsTG-Qa-8ajgCA',
    "ved": "0ahUKEwiUxZ2Hi4j7AhVEY94KHb44CowQ4dUDCAY",
    "uact": 5,
    "oq": query,
    "gs_lcp": "Cgtwcm9kdWN0cy1jYxADMggIABCABBCwAzILCK4BEMoDELADECcyCwiuARDKAxCwAxAnMgsIrgEQygMQsAMQJzILCK4BEMoDELADECdKBAhBGAFQAFgAYMZAaAJwAHgAgAEAiAEAkgEAmAEAyAEFwAEB",
    "sclient": "products-cc"
}
freeDelivery = False
url = 'https://www.google.com/search'

# REQUEST
response = requests.get(url, params=params)

soup = BeautifulSoup(response.text, 'html.parser')

# PARSING RESPONSE
res_dict = {}
product = []
res_data = []

for element in soup.select('div.u30d4'):
    img_container = element.select_one('.eUQRje')
    content_container = element.select_one('.P8xhZc')
    if img_container is not None and content_container is not None:
        product_title = content_container.select_one('.rgHvZc').text
        product_link = re.sub(r"%.*?$", '', unicodedata.normalize("NFKD", re.search(
            "(?P<url>(https|http)?://[^\s]+)", content_container.select_one('.rgHvZc').select_one('a')['href']).group("url")))
        product_origin = ""
        product_price = ""
        product_price_text = ""
        product_reviews = ""
        product_source = img_container.select_one('img')["src"]
        for element_sub_content in content_container.select('div.dD8iuc'):
            if element_sub_content.get("class").count("d1BlKc") > 0:
                product_reviews = unicodedata.normalize(
                    "NFKD", element_sub_content.select_one('div.DApVsf').text).strip()
            else:
                for element_price in element_sub_content.contents:
                    if isinstance(element_price, str):
                        product_origin = element_price.strip()
                    elif not isinstance(element_price, str) and element_price.get("class").count("HRLxBb") > 0:
                        product_price = int(re.sub(r"\.", '', re.sub(
                            r" .*?$", '', unicodedata.normalize("NFKD", element_price.text))))
                        product_price_text = unicodedata.normalize(
                            "NFKD", element_price.text)
        try:
            product_shipping = unicodedata.normalize(
                "NFKD", content_container.select_one('span.dD8iuc').text).strip()
        except:
            product_shipping = ""

        product.append({
            'title': product_title,
            'link': product_link,
            'price': product_price,
            "origin": product_origin,
            'price-text': product_price_text,
            'source': product_source,
            'shipping': product_shipping
        })

res_dict.update({"res_data": product})
print(res_dict)

# for shopping_result in soup.select('.sh-dgr__content'):
#     title = shopping_result.select_one('.Lq5OHe.eaGTj h4').text
#     product_link = f"https://www.google.com{shopping_result.select_one('.Lq5OHe.eaGTj')['href']}"
#     source = shopping_result.select_one('.IuHnof').text
#     price = shopping_result.select_one('span.kHxwFf span').text

#     try:
#         rating = shopping_result.select_one('.Rsc7Yb').text
#     except:
#         rating = None

#     try:
#         reviews = shopping_result.select_one(
#             '.Rsc7Yb').next_sibling.next_sibling
#     except:
#         reviews = None

#     try:
#         delivery = shopping_result.select_one('.vEjMR').text
#     except:
#         delivery = None

#     res_data.append({
#         'title': title,
#         'link': product_link,
#         'source': source,
#         'price': price,
#         'rating': rating,
#         'reviews': reviews,
#         'delivery': delivery,
#     })

# res_dict.update({"res_data": res_data})

# C = (res_dict,
#      f'{params["q"]}{" with free delivery" if freeDelivery else ""}')

# print("Done!")
