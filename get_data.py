from datetime import date

import gdown

date_today = date.today().strftime("%Y%m%d")

url = "https://drive.google.com/uc?id=1GVjIuDprVRDS_z56DO3laWGNGEfuSPmG"
# gdown.download(url , quiet=False)

output = "T20I_ball_by_ball_updated.csv"

# file_dir = "/home/ubuntu/fun/lens/backend/" + output

# out_dir = "/home/ubuntu/fun/lens/backend/CNN_assignment"

# print(file_dir)
# print(out_dir)

# import os
# from zipfile import ZipFile

# if os.path.exists(out_dir):
#     print("File already zipped")
# else:
#     os.makedirs(out_dir)
#     # loading the temp.zip and creating a zip object
#     with ZipFile(file_dir, "r") as zObject:
#         # Extracting all the members of the zip
#         # into a specific location.
#         zObject.extractall(path=out_dir)
