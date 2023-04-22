#conda version
conda 22.11.1


#create conda
conda create -n python_BigDRP python=3.8 anaconda


#activate conda
conda activate python_BigDRP



#install pip
conda install pip


#install requirements
pip install -r requirements.txt


pip install --pre dgl -f https://data.dgl.ai/wheels/cu113/repo.html
pip install --pre dglgo -f https://data.dgl.ai/wheels-test/repo.html
pip install git+https://github.com/ECP-CANDLE/candle_lib@0d32c6bb97ace0370074194943dbeaf9019e6503
