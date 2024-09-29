#%% Imports -------------------------------------------------------------------

import urllib
from pathlib import Path
from configparser import ConfigParser

#%% Inputs --------------------------------------------------------------------

# Paths
root_path = Path(__file__).resolve().parents[1]
utils_path = root_path / "utils"

#%% Functions -----------------------------------------------------------------

def format_dependencies(item):
    dependency_list = []
    for key, value in config.items(item):
        if key:            
            if "conda" in item: 
                ident = " " * 2
                if value: dependency_list.append(f"\n{ident}- {key}={value}")
                else: dependency_list.append(f"\n{ident}- {key}")            
            if "pip" in item:
                ident = " " * 4
                if value: dependency_list.append(f"\n{ident}- {key}={value}")
                else: dependency_list.append(f"\n{ident}- {key}")
        else: dependency_list = ""
    return dependency_list

def update_environment(path):
    
    # Format dependencies
    conda_core = format_dependencies("conda_core")
    conda_spec = format_dependencies("conda_spec")
    pip_core = format_dependencies("pip_core")
    pip_spec = format_dependencies("pip_spec")
    if env_type == "tensorflow":
        conda_tf_gpu = format_dependencies("conda_tf_gpu")
        conda_tf_nogpu = format_dependencies("conda_tf_nogpu")
        pip_tf_gpu = format_dependencies("pip_tf_gpu")
        pip_tf_nogpu = format_dependencies("pip_tf_nogpu")
    
    with open(path, "r") as file:
        environment = file.read()
    environment = environment.replace("{{ env_name }}", "".join(env_name))
    environment = environment.replace("{{ conda_core }}", "".join(conda_core))
    environment = environment.replace("{{ conda_spec }}", "".join(conda_spec))
    environment = environment.replace("{{ pip_core }}", "".join(pip_core))
    environment = environment.replace("{{ pip_spec }}", "".join(pip_spec))
    if env_type == "tensorflow":
        environment = environment.replace("{{ conda_tf_gpu }}", "".join(conda_tf_gpu))
        environment = environment.replace("{{ conda_tf_nogpu }}", "".join(conda_tf_nogpu))
        environment = environment.replace("{{ pip_tf_gpu }}", "".join(pip_tf_gpu))
        environment = environment.replace("{{ pip_tf_nogpu }}", "".join(pip_tf_nogpu))
    
    return environment

def update_install(path):
    with open(path, "r") as file:
        install = file.read()
    install = install.replace("{{ env_name }}", env_name)
    return install

def update_main(path):
    with open(path, "r") as file:
        main = file.read()
    main = main.replace("img src='", "img src='utils/")
    return main

def update_template(path):
    
    with open(path, "r") as file:
        template = file.read()
    with open(utils_path / "README_comments.md", "r") as file:
        comments = file.read()
    template = template.replace("{{ python_version }}", python_version)
    template = template.replace("{{ author }}", author)
    template = template.replace("{{ created }}", created)
    template = template.replace("{{ license }}", license)
    template = template.replace("{{ repo_name }}", repo_name)
    template = template.replace("{{ description }}", description)
    template = template.replace("{{ install }}", install)
    template = template.replace("{{ main }}", main)
    template = template.replace("{{ comments }}", comments)
    if env_type == "tensorflow":
        template = template.replace("{{ tf_version }}", tf_version)
        template = template.replace("{{ cuda_version }}", cuda_version)
        template = template.replace("{{ cudnn_version }}", cudnn_version)
        
    # Index
    sections = []
    for line in template.split("\n"):
        if line.startswith("## "):
            title = line.replace("## ", "").rstrip()
            link = "#" + title.replace(" ", "-").lower()
            sections.append(f"- [{title}]({link})")
    index = "## Index"
    for section in sections:
        index = index + f"\n{section}"
    template = template.replace("{{ index }}", index)   
    
    return template

#%% Initialize ----------------------------------------------------------------

# Parse INI config file
config = ConfigParser()
config.read(utils_path / "config.ini")
repo_name = root_path.name
env_name = config["environment"]["name"]
env_type = config["environment"]["type"]
python_version = config["conda_core"]["python"]
author = config["repository"]["author"]
author = urllib.parse.quote(author)
created = config["repository"]["created"].replace("-", "--")
created = urllib.parse.quote(created)
license = config["repository"]["license"]
license = urllib.parse.quote(license)
description = config["repository"]["description"]
if env_type == "tensorflow":
    tf_version = config["pip_tf_gpu"]["tensorflow-gpu"][1:]
    cuda_version = config["conda_tf_gpu"]["cudatoolkit"]
    cudnn_version = config["conda_tf_gpu"]["cudnn"]

#%% Execute -------------------------------------------------------------------

# Remove preexisting files
for path in list(root_path.glob("*environment*")):
    path.unlink()
for path in list(root_path.glob("*README*")):
    path.unlink()
  
# Update files
if env_type == "base":
    environment = update_environment(utils_path / "environment.yml")
    main = update_main(utils_path / "README_main.md")
    install = update_install(utils_path / "README_install.md")
    template = update_template(utils_path / "README_template.md")
elif env_type == "tensorflow":
    environment_tf_gpu = update_environment(utils_path / "environment_tf_gpu.yml")
    environment_tf_nogpu = update_environment(utils_path / "environment_tf_nogpu.yml")
    main = update_main(utils_path / "README_main.md")
    install = update_install(utils_path / "README_install_tf.md")
    template = update_template(utils_path / "README_template_tf.md")

# Save files
if env_type == "base":
    with open(Path(root_path / "environment.yml"), "w") as file:
        file.write(environment)
elif env_type == "tensorflow":
    with open(Path(root_path / "environment_tf_gpu.yml"), "w") as file:
        file.write(environment_tf_gpu)
    with open(Path(root_path / "environment_tf_nogpu.yml"), "w") as file:
        file.write(environment_tf_nogpu)
with open(Path(root_path / "README.md"), "w") as file:
    file.write(template)
     
