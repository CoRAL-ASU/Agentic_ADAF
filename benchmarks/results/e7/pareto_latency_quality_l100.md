### Latency-Quality Sweep

| config | mode | max_rounds | accuracy | mean latency (s) | p50 (s) | p90 (s) | pareto |
|---|---|---:|---:|---:|---:|---:|---|
| mmqa_full_seq_r6_l100.json | sequential | 6 | 0.090 | 0.000110 | 0.000105 | 0.000141 | yes |
| mmqa_full_seq_r2_l100.json | sequential | 2 | 0.090 | 0.000114 | 0.000105 | 0.000147 | no |
| mmqa_full_seq_r8_l100.json | sequential | 8 | 0.090 | 0.000115 | 0.000108 | 0.000152 | no |
| mmqa_full_seq_r4_l100.json | sequential | 4 | 0.090 | 0.000117 | 0.000109 | 0.000143 | no |
| mmqa_full_par_r8_l100.json | parallel | 8 | 0.090 | 0.000255 | 0.000247 | 0.000293 | no |
| mmqa_full_par_r2_l100.json | parallel | 2 | 0.090 | 0.000265 | 0.000254 | 0.000307 | no |
| mmqa_full_par_r6_l100.json | parallel | 6 | 0.090 | 0.000270 | 0.000259 | 0.000321 | no |
| mmqa_full_par_r4_l100.json | parallel | 4 | 0.090 | 0.000272 | 0.000264 | 0.000315 | no |

### Pareto Frontier

| config | mode | max_rounds | accuracy | mean latency (s) |
|---|---|---:|---:|---:|
| mmqa_full_seq_r6_l100.json | sequential | 6 | 0.090 | 0.000110 |
