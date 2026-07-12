1- Run install_packages.bat\
2- The data is already preprocessed, but to restart the preprocessing again run the following command from the same folder:\
```python scripts/preprocess.py --input data/raw/labor_law.txt```\
3- In a terminal in the main folder run the following command:\
```python preflight.py``` \
to make sure that all neccessary packages, data is preprocessed and available and that ollama is set up\
4- enter the systems directory and run the following command to start the experiments:\
```python run_experiments.py``` \
you can include a parameter named ```--systems``` if you only want to run a couple of the experiments and not all, for example\
```python run_experiments.py --systems bm25_fixed hybrid_fixed``` \
5- After the experiment finishes run the following command to evaluate the findings:\
```python evaluate.py``` 
