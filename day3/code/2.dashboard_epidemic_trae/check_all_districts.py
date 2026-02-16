import json

with open('static/hksar_18_district_boundary.json', 'r') as f:
    data = json.load(f)
    print("GeoJSON 中的地区名称列表:")
    for feature in data['features']:
        print(f"'{feature['properties'].get('地區')}'")
