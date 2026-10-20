# V2 实验原始报告（自动生成）

- 配置数: 7
- 种子: 200..205（6 个）
- 步数: 6000（300 世界秒）
- 总运行: 42
- 墙钟: 554.2s
- 感知刷新 stride: 8

## 每配置汇总（均值 ± 标准差 [min, max]）

| 配置 | tech_verified | facilities_active | alive_count | total_agents_created | innovations | goal_completion_rate | cooperation_index | visual_active_dims | lowlevel_mean_reward | 灭绝率 | 上限违规 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| c_bc_vision_off | 4.17±0.90 [3.00,5.00] | 0.00±0.00 [0.00,0.00] | 8.00±0.00 [8.00,8.00] | 8.00±0.00 [8.00,8.00] | 0.00±0.00 [0.00,0.00] | 0.96±0.01 [0.95,0.97] | 0.00±0.00 [0.00,0.00] | 0.00±0.00 [0.00,0.00] | 0.00±0.00 [0.00,0.00] | 0.00 | 0 |
| c_bc_vision_on | 4.17±0.90 [3.00,5.00] | 0.00±0.00 [0.00,0.00] | 8.00±0.00 [8.00,8.00] | 8.00±0.00 [8.00,8.00] | 0.00±0.00 [0.00,0.00] | 0.96±0.01 [0.95,0.97] | 0.00±0.00 [0.00,0.00] | 16.00±0.00 [16.00,16.00] | 0.00±0.00 [0.00,0.00] | 0.00 | 0 |
| c_rl_policy | 4.17±1.21 [3.00,6.00] | 0.00±0.00 [0.00,0.00] | 8.00±0.00 [8.00,8.00] | 8.00±0.00 [8.00,8.00] | 0.00±0.00 [0.00,0.00] | 0.97±0.01 [0.95,0.97] | 0.00±0.00 [0.00,0.00] | 15.67±0.75 [14.00,16.00] | 0.00±0.00 [0.00,0.00] | 0.00 | 0 |
| c_rl_random | 4.00±1.00 [3.00,5.00] | 0.00±0.00 [0.00,0.00] | 8.00±0.00 [8.00,8.00] | 8.00±0.00 [8.00,8.00] | 0.00±0.00 [0.00,0.00] | 0.95±0.02 [0.90,0.97] | 0.00±0.00 [0.00,0.00] | 14.00±0.00 [14.00,14.00] | 0.00±0.00 [0.00,0.00] | 0.00 | 0 |
| cap32_finite | 4.33±0.94 [3.00,5.00] | 0.00±0.00 [0.00,0.00] | 32.00±0.00 [32.00,32.00] | 32.00±0.00 [32.00,32.00] | 0.00±0.00 [0.00,0.00] | 0.93±0.01 [0.92,0.93] | 0.00±0.00 [0.00,0.00] | 16.00±0.00 [16.00,16.00] | 0.00±0.00 [0.00,0.00] | 0.00 | 0 |
| cap32_infinite | 4.00±1.00 [3.00,5.00] | 0.00±0.00 [0.00,0.00] | 29.83±2.73 [26.00,32.00] | 29.83±2.73 [26.00,32.00] | 0.00±0.00 [0.00,0.00] | 0.88±0.08 [0.70,0.94] | 0.00±0.00 [0.00,0.00] | 16.00±0.00 [16.00,16.00] | 0.00±0.00 [0.00,0.00] | 0.00 | 0 |
| cap8_infinite | 4.00±1.00 [3.00,5.00] | 0.00±0.00 [0.00,0.00] | 8.00±0.00 [8.00,8.00] | 8.00±0.00 [8.00,8.00] | 0.00±0.00 [0.00,0.00] | 0.93±0.06 [0.79,0.98] | 0.00±0.00 [0.00,0.00] | 16.00±0.00 [16.00,16.00] | 0.00±0.00 [0.00,0.00] | 0.00 | 0 |

## 配对差（同 seed，95% CI）

- 感知开 vs 关 :: tech_verified: {'mean': 0.0, 'low': 0.0, 'high': 0.0, 'wins': 0, 'losses': 0, 'ties': 6, 'n': 6}
- 感知开 vs 关 :: facilities_active: {'mean': 0.0, 'low': 0.0, 'high': 0.0, 'wins': 0, 'losses': 0, 'ties': 6, 'n': 6}
- 感知开 vs 关 :: alive_count: {'mean': 0.0, 'low': 0.0, 'high': 0.0, 'wins': 0, 'losses': 0, 'ties': 6, 'n': 6}
- 感知开 vs 关 :: goal_completion_rate: {'mean': 0.0, 'low': 0.0, 'high': 0.0, 'wins': 0, 'losses': 0, 'ties': 6, 'n': 6}
- 感知开 vs 关 :: visual_active_dims: {'mean': 16.0, 'low': 16.0, 'high': 16.0, 'wins': 6, 'losses': 0, 'ties': 0, 'n': 6}
- 感知开 vs 关 :: cooperation_index: {'mean': 0.0, 'low': 0.0, 'high': 0.0, 'wins': 0, 'losses': 0, 'ties': 6, 'n': 6}
- RL 低层 vs BC :: tech_verified: {'mean': 0.0, 'low': -1.0496, 'high': 1.0496, 'wins': 2, 'losses': 1, 'ties': 3, 'n': 6}
- RL 低层 vs BC :: facilities_active: {'mean': 0.0, 'low': 0.0, 'high': 0.0, 'wins': 0, 'losses': 0, 'ties': 6, 'n': 6}
- RL 低层 vs BC :: alive_count: {'mean': 0.0, 'low': 0.0, 'high': 0.0, 'wins': 0, 'losses': 0, 'ties': 6, 'n': 6}
- RL 低层 vs BC :: goal_completion_rate: {'mean': 0.0041, 'low': -0.0076, 'high': 0.0157, 'wins': 3, 'losses': 3, 'ties': 0, 'n': 6}
- RL 低层 vs BC :: visual_active_dims: {'mean': -0.3333, 'low': -1.1157, 'high': 0.449, 'wins': 0, 'losses': 1, 'ties': 5, 'n': 6}
- RL 低层 vs BC :: cooperation_index: {'mean': 0.0, 'low': 0.0, 'high': 0.0, 'wins': 0, 'losses': 0, 'ties': 6, 'n': 6}
- 随机低层 vs BC :: tech_verified: {'mean': -0.1667, 'low': -1.1087, 'high': 0.7754, 'wins': 1, 'losses': 1, 'ties': 4, 'n': 6}
- 随机低层 vs BC :: facilities_active: {'mean': 0.0, 'low': 0.0, 'high': 0.0, 'wins': 0, 'losses': 0, 'ties': 6, 'n': 6}
- 随机低层 vs BC :: alive_count: {'mean': 0.0, 'low': 0.0, 'high': 0.0, 'wins': 0, 'losses': 0, 'ties': 6, 'n': 6}
- 随机低层 vs BC :: goal_completion_rate: {'mean': -0.0147, 'low': -0.0416, 'high': 0.0122, 'wins': 2, 'losses': 4, 'ties': 0, 'n': 6}
- 随机低层 vs BC :: visual_active_dims: {'mean': -2.0, 'low': -2.0, 'high': -2.0, 'wins': 0, 'losses': 6, 'ties': 0, 'n': 6}
- 随机低层 vs BC :: cooperation_index: {'mean': 0.0, 'low': 0.0, 'high': 0.0, 'wins': 0, 'losses': 0, 'ties': 6, 'n': 6}
- 32 人上限 vs 8 人（无限世界） :: tech_verified: {'mean': 0.0, 'low': 0.0, 'high': 0.0, 'wins': 0, 'losses': 0, 'ties': 6, 'n': 6}
- 32 人上限 vs 8 人（无限世界） :: facilities_active: {'mean': 0.0, 'low': 0.0, 'high': 0.0, 'wins': 0, 'losses': 0, 'ties': 6, 'n': 6}
- 32 人上限 vs 8 人（无限世界） :: alive_count: {'mean': 21.8333, 'low': 18.9642, 'high': 24.7025, 'wins': 6, 'losses': 0, 'ties': 0, 'n': 6}
- 32 人上限 vs 8 人（无限世界） :: goal_completion_rate: {'mean': -0.045, 'low': -0.0701, 'high': -0.0199, 'wins': 0, 'losses': 6, 'ties': 0, 'n': 6}
- 32 人上限 vs 8 人（无限世界） :: visual_active_dims: {'mean': 0.0, 'low': 0.0, 'high': 0.0, 'wins': 0, 'losses': 0, 'ties': 6, 'n': 6}
- 32 人上限 vs 8 人（无限世界） :: cooperation_index: {'mean': 0.0, 'low': 0.0, 'high': 0.0, 'wins': 0, 'losses': 0, 'ties': 6, 'n': 6}
- 32 人上限（有限世界） vs 8 人（无限世界） :: tech_verified: {'mean': 0.3333, 'low': -0.449, 'high': 1.1157, 'wins': 1, 'losses': 0, 'ties': 5, 'n': 6}
- 32 人上限（有限世界） vs 8 人（无限世界） :: facilities_active: {'mean': 0.0, 'low': 0.0, 'high': 0.0, 'wins': 0, 'losses': 0, 'ties': 6, 'n': 6}
- 32 人上限（有限世界） vs 8 人（无限世界） :: alive_count: {'mean': 24.0, 'low': 24.0, 'high': 24.0, 'wins': 6, 'losses': 0, 'ties': 0, 'n': 6}
- 32 人上限（有限世界） vs 8 人（无限世界） :: goal_completion_rate: {'mean': -0.0025, 'low': -0.0712, 'high': 0.0662, 'wins': 1, 'losses': 5, 'ties': 0, 'n': 6}
- 32 人上限（有限世界） vs 8 人（无限世界） :: visual_active_dims: {'mean': 0.0, 'low': 0.0, 'high': 0.0, 'wins': 0, 'losses': 0, 'ties': 6, 'n': 6}
- 32 人上限（有限世界） vs 8 人（无限世界） :: cooperation_index: {'mean': 0.0, 'low': 0.0, 'high': 0.0, 'wins': 0, 'losses': 0, 'ties': 6, 'n': 6}

## 逐 run

| config | seed | tech | fac | alive | created | extinct | cause | visual_dims | wall_s |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| c_bc_vision_off | 200 | 3 | 0 | 8 | 8 | False |  | 0 | 5.725 |
| c_bc_vision_off | 201 | 5 | 0 | 8 | 8 | False |  | 0 | 5.956 |
| c_bc_vision_off | 202 | 5 | 0 | 8 | 8 | False |  | 0 | 5.429 |
| c_bc_vision_off | 203 | 5 | 0 | 8 | 8 | False |  | 0 | 5.512 |
| c_bc_vision_off | 204 | 3 | 0 | 8 | 8 | False |  | 0 | 5.709 |
| c_bc_vision_off | 205 | 4 | 0 | 8 | 8 | False |  | 0 | 5.812 |
| c_bc_vision_on | 200 | 3 | 0 | 8 | 8 | False |  | 16 | 11.544 |
| c_bc_vision_on | 201 | 5 | 0 | 8 | 8 | False |  | 16 | 11.834 |
| c_bc_vision_on | 202 | 5 | 0 | 8 | 8 | False |  | 16 | 11.479 |
| c_bc_vision_on | 203 | 5 | 0 | 8 | 8 | False |  | 16 | 11.203 |
| c_bc_vision_on | 204 | 3 | 0 | 8 | 8 | False |  | 16 | 11.624 |
| c_bc_vision_on | 205 | 4 | 0 | 8 | 8 | False |  | 16 | 11.529 |
| c_rl_policy | 200 | 3 | 0 | 8 | 8 | False |  | 16 | 12.244 |
| c_rl_policy | 201 | 3 | 0 | 8 | 8 | False |  | 16 | 12.661 |
| c_rl_policy | 202 | 6 | 0 | 8 | 8 | False |  | 16 | 13.031 |
| c_rl_policy | 203 | 5 | 0 | 8 | 8 | False |  | 16 | 11.89 |
| c_rl_policy | 204 | 3 | 0 | 8 | 8 | False |  | 16 | 12.691 |
| c_rl_policy | 205 | 5 | 0 | 8 | 8 | False |  | 14 | 12.932 |
| c_rl_random | 200 | 3 | 0 | 8 | 8 | False |  | 14 | 7.238 |
| c_rl_random | 201 | 3 | 0 | 8 | 8 | False |  | 14 | 6.833 |
| c_rl_random | 202 | 5 | 0 | 8 | 8 | False |  | 14 | 7.878 |
| c_rl_random | 203 | 5 | 0 | 8 | 8 | False |  | 14 | 7.59 |
| c_rl_random | 204 | 3 | 0 | 8 | 8 | False |  | 14 | 7.77 |
| c_rl_random | 205 | 5 | 0 | 8 | 8 | False |  | 14 | 7.768 |
| cap8_infinite | 200 | 3 | 0 | 8 | 8 | False |  | 16 | 12.006 |
| cap8_infinite | 201 | 3 | 0 | 8 | 8 | False |  | 16 | 13.94 |
| cap8_infinite | 202 | 5 | 0 | 8 | 8 | False |  | 16 | 14.73 |
| cap8_infinite | 203 | 5 | 0 | 8 | 8 | False |  | 16 | 13.274 |
| cap8_infinite | 204 | 3 | 0 | 8 | 8 | False |  | 16 | 12.787 |
| cap8_infinite | 205 | 5 | 0 | 8 | 8 | False |  | 16 | 13.277 |
| cap32_infinite | 200 | 3 | 0 | 26 | 26 | False |  | 16 | 17.949 |
| cap32_infinite | 201 | 3 | 0 | 32 | 32 | False |  | 16 | 22.728 |
| cap32_infinite | 202 | 5 | 0 | 31 | 31 | False |  | 16 | 22.446 |
| cap32_infinite | 203 | 5 | 0 | 32 | 32 | False |  | 16 | 22.649 |
| cap32_infinite | 204 | 3 | 0 | 26 | 26 | False |  | 16 | 16.921 |
| cap32_infinite | 205 | 5 | 0 | 32 | 32 | False |  | 16 | 22.093 |
| cap32_finite | 200 | 3 | 0 | 32 | 32 | False |  | 16 | 20.174 |
| cap32_finite | 201 | 5 | 0 | 32 | 32 | False |  | 16 | 21.193 |
| cap32_finite | 202 | 5 | 0 | 32 | 32 | False |  | 16 | 20.72 |
| cap32_finite | 203 | 5 | 0 | 32 | 32 | False |  | 16 | 21.153 |
| cap32_finite | 204 | 3 | 0 | 32 | 32 | False |  | 16 | 20.561 |
| cap32_finite | 205 | 5 | 0 | 32 | 32 | False |  | 16 | 21.522 |
