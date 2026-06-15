import zipfile
import os

merged_zip = 'dataset/full_dataset.zip'
output_dir = './data/vggface2_hq_extracted'

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

print("Extracting merged ZIP... this will work now!")
with zipfile.ZipFile(merged_zip, 'r') as zf:
    zf.extractall(output_dir)
print("Extraction complete.")