import requests
from requests.auth import HTTPBasicAuth
import json
import os


# page_ids = ["2483290456"]  # , "<page_id_2>", "<page_id_3"]
space_key = "~712020d26a0bf843a04a54b0e4c6254eb599ec"


# Your Confluence domain
domain = "https://bouncybear.atlassian.net"

# The ID of the page you want to retrieve
page_id = "393222"

# Your Atlassian username
username = os.getenv('CONFLUENCE_USERNAME')

# Your Atlassian API token
api_token = os.getenv('CONFLUENCE_API_TOKEN')

# Construct the URL to access the Confluence API
url = f"{domain}/wiki/rest/api/content/{page_id}?expand=body.storage"

# Make the GET request to Confluence API
response = requests.get(url, auth=HTTPBasicAuth(username, api_token))

# Check if the request was successful
if response.status_code == 200:
    # Parse the JSON response
    data = response.json()

    # Extract the page content in storage format
    content = data['body']['storage']['value']

    print("Content Retrieved Successfully:")
    print(content)
else:
    print("Failed to retrieve content:")
    print("Status Code:", response.status_code)
    print("Response:", response.text)


from atlassian import Confluence

# Initialize the Confluence instance
confluence = Confluence(
    url=domain,
    username=username,
    password=api_token,
    cloud=True # Set to True if you are using Confluence Cloud
)

# Get page content
content = confluence.get_page_by_id(page_id, expand='body.storage')

# Print page title and content
print("Title:", content['title'])
print("Content:", content['body']['storage']['value'])

