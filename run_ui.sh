#!/bin/bash
cd ~/dev/btc_alpha || exit
source venv/bin/activate 2>/dev/null
python ui/app.py &
sleep 2
cmd.exe /C start http://localhost:8080
wait
