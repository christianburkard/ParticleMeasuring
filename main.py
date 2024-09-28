#%% Imports -------------------------------------------------------------------

from pathlib import Path

# Functions
from functions import process

#%% Inputs --------------------------------------------------------------------

# Paths
data_path = Path(Path.cwd(), 'data', 'Test')
model_name = "model-weights_0512.h5"
paths = list(data_path.glob("**/*.jpg"))

# Patches
size = int(model_name[14:18])
overlap = size // 4 # overlap between patches

# Rescale factor
rf = 0.5

#%% Execute -------------------------------------------------------------------

def capsSegMain(filePath, pix_to_um):
    process(filePath, size, overlap, pix_to_um, model_name, rf=rf, save=True)

if __name__ == "__main__":
    for i, path in enumerate(paths):
        # if i == 0:
        process(path, size, overlap, model_name, rf=rf, save=True)