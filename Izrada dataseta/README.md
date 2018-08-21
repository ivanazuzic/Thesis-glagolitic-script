#Instalacija environmenta

set PATH=C:\ProgramData\Anaconda3\Scripts;%PATH%
conda create -n <imeENVa> pip python=3.5 
activate <imeENVa>
pip install --upgrade tensorflow-gpu
conda install -c numba cudatoolkit 
conda install -c conda-forge keras    (pip install --upgrade --no-deps keras) (conda install -c conda-forge --no-deps keras  <-- dobro)
conda install spyder
conda install -c anaconda ipython 
conda install jupyter
conda install scikit-learn

conda install -c conda-forge matplotlib
conda install pillow
conda install matplotlib
conda install -c menpo opencv
conda install scikit-learn

#Pokretanje
set PATH=C:\ProgramData\Anaconda3\Scripts;%PATH%
activate <imeENVa>
set PATH=%PATH%;D:\Users\izuzic\CUDA\v9.0\bin;D:\Users\izuzic\CUDA\v9.0\include;D:\Users\izuzic\CUDA\v9.0\lib\x64
set "KERAS_BACKEND=tensorflow"
cd <lokacija repozitorija>