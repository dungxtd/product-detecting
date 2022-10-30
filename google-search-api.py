# Import the beautifulsoup
# and request libraries of python.
import requests
import bs4
import re
# Make two strings with default google search URL
# 'https://google.com/search?q=' and
# our customized search keyword.
# Concatenate them
url = 'https://google.com/search'
params = {
    "q": "bánh gạo ichi",
    "tbm": "shop",
    "sxsrf": "ALiCzsbStIlArlzVUTwd2PBUqqTqIwGsFw%3A1667121559775",
    "ei": "l0FeY9jGLoXS2roP_oy-0Ag",
    "ved": "0ahUKEwiY7vfiz4f7AhUFqVYBHX6GD4oQ4dUDCAY",
    "uact": 5,
    "oq": "bánh gạo ichi",
    "gs_lcp": "Cgtwcm9kdWN0cy1jYxADMgcIIxCwAxAnMggIABCABBCwAzIICAAQgAQQsAMyCAgAEIAEELADMggIABCABBCwAzIICAAQgAQQsAMyCAgAEIAEELADMggIABCABBCwAzIJCAAQBxAeELADMgkIABAHEB4QsAMyCwiuARDKAxCwAxAnSgQIQRgBUABYAGCaU2gCcAB4AIABAIgBAJIBAJgBAMgBC8ABAQ",
    "sclient": "products-cc"
}

# Fetch the URL data using requests.get(url),
# store it in a variable, request_result.
request_result = requests.get(url, params)

# Creating soup from the fetched request
soup = bs4.BeautifulSoup(request_result.text, "html.parser")
heading_object = soup.find_all('div', {"class": "rgHvZc"})

# Iterate through the object
# and print it as a string.
for info in heading_object:
    children = info.findChildren("a", href=True)
    print(info.getText())
    print(
        re.search("(?P<url>(https|http)?://[^\s]+)", children[0]['href']).group("url"))
    print("------")
