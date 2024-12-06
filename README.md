# Tetris-RL-AI
Reinforcement Learning model for Tetris

## Commands to run Tetris
Compile:
```
javac -cp "./lib/*:." @tetris.srcs
```
Run:
```
java -cp "./lib/*:." edu.bu.tetris.Main -p 5000 -t 200 -v 100 -g 0.99 -n 1e-5 | tee my_logfile.log
```

