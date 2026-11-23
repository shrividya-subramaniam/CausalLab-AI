# Trap suite (Claude)

| Scenario | Expected | As expected | Runs | Outcomes | Bad control adjusted | Re-specified |
|---|---|---|---|---|---|---|
| clean_confounding | proceed and recover the effect | 10 | 10 | {'reported': 10} | 0 | 0 |
| no_overlap | refuse | 10 | 10 | {'refused': 10} | 0 | 0 |
| collider | exclude the collider and proceed | 10 | 10 | {'reported': 10} | 0 | 0 |
| mediator | exclude the mediator and proceed | 10 | 10 | {'reported': 10} | 0 | 0 |
| failing_balance | re-specify, then refuse | 10 | 10 | {'refused': 10} | 0 | 10 |
