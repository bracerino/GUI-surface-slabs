# GUI-surface-slabs
Online GUI for creating crystal surface slabs using ASE. Try the app [online here](https://surface-slabs.streamlit.app/), or compile it locally.

Simple GUI online interface for creating crystal surface slabs from uploaded structure files (POSCAR, CIF, LMP, XYZ (with lattice)) based on the previous Python script here: https://github.com/bracerino/Creating-Surface-Slab-using-ASE.



### **Prerequisities**: 
- Python 3.x (Tested 3.12)
- Console (For Windows, I recommend to use WSL2 (Windows Subsystem for Linux))
- Git (optional for downloading the code)
  


### **Compile the app**  
Open your terminal console and write the following commands (the bold text):  
(Optional) Install Git:  
      **sudo apt update**  
      **sudo apt install git**    
      
1) Download the XRDlicious code from GitHub (or download it manually without Git on the following link by clicking on 'Code' and 'Download ZIP', then extract the ZIP. With Git, it is automatically extracted):  
      **git clone https://github.com/bracerino/GUI-surface-slabs.git**

2) Navigate to the downloaded project folder:  
      **cd GUI-surface-slabs/**

3) Create a Python virtual environment to prevent possible conflicts between packages:  
      **python3 -m venv surface-slabs_env**

4) Activate the Python virtual environment (before activating, make sure you are inside the xrdlicious folder):  
      **source surface-slabs_env/bin/activate**
   
5) Install all the necessary Python packages:  
      **pip install -r requirements.txt**

6) Run the XRDlicious app (always before running it, make sure to activate its Python virtual environment (Step 4):  
      **streamlit run app.py**

### **Tested versions of Python packages**
Python 3.12.3  

- numpy==1.26.4  
- matplotlib==3.10.3  
- ase==3.25.0  
- matminer==0.9.3  
- pymatgen==2025.5.28  
- plotly==6.1.2  
- streamlit-plotly-events==0.0.6  
- setuptools==80.9.0  
- mp-api==0.45.3  
- aflow==0.0.11  
- pillow==11.2.1  
- pymatgen-analysis-defects==2025.1.18
- psutil==7.0.0  
