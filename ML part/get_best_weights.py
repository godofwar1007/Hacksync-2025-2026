import os
import shutil
from glob import glob

print("Backing up the model to Drive...")

# searching for the best.pt file in the runs folder
files = glob('/content/runs/**/best.pt', recursive=True)

if files:
    # grab the latest one just in case there are multiple
    best_weights = max(files, key=os.path.getctime)
    
    # copying it to my drive
    destination = '/content/drive/MyDrive/accident_model_final.pt'
    shutil.copy(best_weights, destination)
    
    print(f"Done! Saved the model to: {destination}")
    print("Safe from disconnects now.")

else:
    print("Couldn't find best.pt, checking for last.pt instead...")
    
    # fallback: try to find the last checkpoint if best isn't there
    backup_files = glob('/content/runs/**/last.pt', recursive=True)
    
    if backup_files:
        last_weights = max(backup_files, key=os.path.getctime)
        shutil.copy(last_weights, '/content/drive/MyDrive/accident_model_backup.pt')
        print("Saved last.pt as a backup.")
    else:
        print("Error: Can't find any model files at all. Check the folder.")
