import json

with open('static/hksar_18_district_boundary.json', 'r') as f:
    data = json.load(f)
    if data['features']:
        print(data['features'][0]['properties'])
