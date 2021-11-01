nohup python deberta.py models/tk_vs1000_frozen.json 0 > logs/vs_1000_deberta.log 2>&1 &
nohup python deberta.py models/tk_vs5000_frozen.json 1 > logs/vs5000_deberta.log 2>&1 &
nohup python deberta.py models/tk_vs10000_frozen.json 2 > logs/vs10000_deberta.log 2>&1 &
