import json

with open("songs/failed_downloads.json", "r", encoding="utf-8") as f:
    fail_link = json.load(f)
with open("data/MIR-CE500_corrected.json", "r", encoding="utf-8") as f:
    data_json = json.load(f)
table = {}
for i in range(len(fail_link)):
    idx = fail_link[i]["index_raw"]
    table[idx] = data_json[str(idx)]
filename = "remain.json"
with open(filename, 'w') as json_file:
    json.dump(table, json_file, indent=4)