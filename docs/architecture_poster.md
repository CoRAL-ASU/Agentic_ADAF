# Planner-free, Log-Mediated QA Architecture

> Collaborative specialist agents coordinate through an append-only log to answer user questions while a verifier safeguards the final response.

```
┌───────────────┐      Read       ┌───────────────────────────────────────┐
│   User Query  │ ──────────────► │            Shared Append Log          │
└───────────────┘                 │  • Agent, Type, Content, Metadata     │
                                  │  • Read/Write history                │
                                  └───────────────────────────────────────┘
                                          ▲                     ▲
                                          │ Append              │ Read & Flag
┌─────────────────────┐        ┌──────────┴──────────┐          │
│ Specialist Agents   │        │  Controller Agent   │          │
│ ┌───────────────┐   │        │  • Turn-taking      │          │
│ │ Table Agent   │   │        │  • Simple scheduler │          │
│ ├───────────────┤   │        │  • Not a planner    │          │
│ │ Context Agent │   │        └──────────┬──────────┘          │
│ ├───────────────┤   │                   │                     │
│ │ Visual Agent  │   │        Append     │                     │
│ └───────────────┘   │                   │                     │
└─────────────────────┘                   ▼                     │
                                  ┌─────────────────────┐      │
                                  │ Summarizing Agent   │      │
                                  │  • Synthesizes      │      │
                                  │    candidate answer │      │
                                  └──────────┬──────────┘      │
                                             │ Answer          │
                                             ▼                 │
                                  ┌─────────────────────┐      │
                                  │ Verification Agent  │ ◄────┘
                                  │  • Cross-checks     │
                                  │    final response   │
                                  │  • Needs-Answer flag│
                                  └──────────┬──────────┘
                                             │ Verified Answer
                                             ▼
                                        ┌───────────────┐
                                        │     User      │
                                        └───────────────┘
```

## Component Roles

- **Specialist Agents** (Table, Context, Visual) extract and reason over domain-specific evidence. Each agent appends observations, intermediate reasoning, or lookup results to the shared log.
- **Controller Agent** orchestrates the conversation by selecting which specialist should act next. It does not plan full solutions; instead it enforces orderly turn-taking and ensures the log progresses.
- **Shared Append Log** is the central communication medium. Every agent reads from this append-only structure to understand context, then writes new entries capturing actions, findings, and requests for follow-up work.
- **Summarizing Agent** monitors the log for a "need answer" flag. When triggered, it synthesizes a comprehensive answer using the accumulated evidence and appends the draft to the log.
- **Verification Agent** reads the summarized answer, cross-checks against log evidence, and either confirms the final response or requests further investigation by writing a verification note back to the log.

## Interaction Flow

1. **User submits a question.** The controller logs the request and selects the appropriate specialist to begin evidence gathering.
2. **Specialists iterate via the log.** Each specialist reads prior entries, performs retrieval or reasoning, then appends their findings. The controller assigns turns until sufficient evidence exists.
3. **Summarizer produces an answer.** Once the log indicates a response is needed, the summarizing agent compiles the reasoning and proposes an answer in the log.
4. **Verifier cross-checks.** The verification agent validates the proposed answer against log contents. If discrepancies exist, it flags the issue and may prompt additional specialist turns.
5. **User receives the verified answer.** When the verification agent approves, the final answer is surfaced to the user along with any supporting rationale captured in the log.

This log-mediated collaboration enables robust multi-step reasoning without relying on a monolithic planner, while maintaining transparency and accountability for every decision taken by the agents.
