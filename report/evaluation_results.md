# Evaluation Results: Chatbot vs ReAct Agent

## Aggregate metrics

| System | Accuracy | Avg tokens | Avg LLM calls | Avg latency | Total cost |
| :--- | :--- | :--- | :--- | :--- | :--- |
| Chatbot | 40% (2/5) | 1647 | 1.0 | 12898 ms | $0.08235 |
| Agent v1 | 60% (3/5) | 1549 | 1.8 | 5695 ms | $0.07747 |
| Agent v2 | 80% (4/5) | 3222 | 2.6 | 9891 ms | $0.16111 |

## Per-question results

| # | Type | Question | Chatbot | Agent v1 | Agent v2 | Winner (cb vs v2) |
| :-- | :-- | :-- | :-- | :-- | :-- | :-- |
| 1 | simple | What year did the show Breaking Bad premiere? | correct | correct | correct | Draw |
| 2 | simple | What is the TVmaze rating of Breaking Bad? | WRONG | correct | correct | Agent |
| 3 | multi | How many total hours would it take to binge-watch all episodes of Breaking Bad? | correct | correct | correct | Draw |
| 4 | multi | Which has a higher TVmaze rating, Breaking Bad or Game of Thrones, and by how much? | WRONG | WRONG | WRONG | Draw |
| 5 | multi | What is the combined number of episodes of Breaking Bad and Stranger Things? | WRONG | WRONG | correct | Agent |

## Sample answers (Agent v2)

- **What year did the show Breaking Bad premiere?**
  - Breaking Bad premiered in 2008.
- **What is the TVmaze rating of Breaking Bad?**
  - The TVmaze rating of Breaking Bad is 9.2.
- **How many total hours would it take to binge-watch all episodes of Breaking Bad?**
  - It would take approximately 62.0 hours to binge-watch all episodes of Breaking Bad.
- **Which has a higher TVmaze rating, Breaking Bad or Game of Thrones, and by how much?**
  - Breaking Bad has a higher TVmaze rating (9.2 vs 9.1), by 0.1 points.
- **What is the combined number of episodes of Breaking Bad and Stranger Things?**
  - 104
