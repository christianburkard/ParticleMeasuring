#%% Imports -------------------------------------------------------------------

import napari
from pathlib import Path

# Functions
from functions import process

#%% Inputs --------------------------------------------------------------------

# Paths
data_path = Path(Path.cwd(), 'data')
image_name = "20240313_E1_0.jpg"
image_path = list(data_path.glob(f"**/*{image_name}"))[0]
model_name = "model-weights_0512.h5"

# Patches
size = int(model_name[14:18])
overlap = size // 4 # overlap between patches

# Rescale factor
rf = 0.5

#%% Execute -------------------------------------------------------------------

outputs = process(image_path, size, overlap, model_name, rf=rf, save=False)
    
#%% Display ------------------------------------------------------------------- 

viewer = napari.Viewer()
viewer.add_image(
    outputs["img"], name="image", contrast_limits=(0, 1), opacity=0.33)
viewer.add_image(
    outputs["sProbs"], name="sProbs", contrast_limits=(0, 1), 
    blending="additive", colormap="yellow", visible=False
    )
viewer.add_image(
    outputs["cProbs"], name="cProbs", contrast_limits=(0, 1), 
    blending="additive", colormap="cyan", visible=False
    )
viewer.add_image(
    outputs["sDisplay"], name="sDisplay", contrast_limits=(0, 255), 
    blending="additive", colormap="yellow"
    )
viewer.add_image(
    outputs["cDisplay"], name="cDisplay", contrast_limits=(0, 255),
    blending="additive", colormap="cyan"
    )