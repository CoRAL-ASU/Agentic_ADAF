# Chain-of-Table Prompt Templates
These examples illustrate how to drive **gpt-3.5-turbo** to perform stepwise reasoning over tables. Each example shows a table, a question, and a chain of operations that the LLM should follow. Feel free to adapt or extend these templates for your own tasks.

---

### Example 1 – Revenue Growth

```
Input Table:
| Company | Revenue 2020 | Revenue 2021 |
|---------|--------------|--------------|
| ABC     | 1,000        | 1,500        |
| XYZ     | 2,000        | 1,800        |

Question: Which company had the largest growth in revenue?
```

**Chain-of-Table Reasoning**
1. Compute `Revenue_Growth = Revenue 2021 - Revenue 2020` for each row.
2. Select the row with the maximum `Revenue_Growth`.
3. Report the company name from that row as the final answer.

Expected Answer: `ABC`

---

### Example 2 – Filtering and Summation

```
Input Table:
| Player | Team | Points |
|--------|------|-------|
| Alice  | Red  | 12    |
| Bob    | Blue | 15    |
| Carol  | Red  | 20    |

Question: How many total points were scored by the Red team?
```

**Chain-of-Table Reasoning**
1. Filter rows where `Team` equals `Red`.
2. Sum the `Points` column of the filtered table.
3. Return that sum.

Expected Answer: `32`

---

### Example 3 – Joining and Computing Ratios

```
Table A:
| CompanyID | Employees |
|-----------|-----------|
| 1         | 50        |
| 2         | 75        |

Table B:
| CompanyID | Revenue |
|-----------|---------|
| 1         | 2,000   |
| 2         | 3,000   |

Question: Which company has the higher revenue per employee?
```

**Chain-of-Table Reasoning**
1. Join Table A and Table B on `CompanyID`.
2. Compute `RevPerEmp = Revenue / Employees` for each joined row.
3. Select the row with the highest `RevPerEmp` and output the `CompanyID`.

Expected Answer: `2`

---

Use these chains as demonstrations when prompting **gpt-3.5-turbo**. The agent code will execute each described operation and track intermediate tables, enabling transparent, step-by-step reasoning.
