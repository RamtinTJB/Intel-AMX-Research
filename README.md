# Intel AMX Research

## Introduction

This repository contains a modified version of [HD-Classification](https://github.com/UCSD-SEELab/HD-Classification/tree/master/CPU) that uses Intel AMX (Advanced Matrix Extensions) accelerators to optimize matrix multiplications.

The file [HD.py](https://github.com/RamtinTJB/Intel-AMX-Research/blob/main/HD-Classification/CPU/HD.py) contains the original code and [HD_amx.py](https://github.com/RamtinTJB/Intel-AMX-Research/blob/main/HD-Classification/CPU/HD_amx.py) contains the AMX optimized version.

## Running the code
```sh
python main.py --path isolet.pickle --alg idlv --size 16384 --amx
```

- An arbitrary dataset size can be chosen using the `--size` command line argument to make benchmarking easier.
- Without the `--amx` argument, the original version will be run

## Verifying AMX Usage
To verify that AMX is being used and to check which cores had an active AMX unit during the execution of our program, we can use the following command:
```sh
perf stat -a --per-core -e exe.amx_busy -- <command to benchmark>
```

## TODO

- [ ] Implement quantized INT8 multiplication in the `max_match` function
- [ ] Initialize all data types as bf16 to avoid conversions and improve performance
