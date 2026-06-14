# LCF shared data contract (lcf/data/)

Produced by `lcf/lcf_impl/lfud_data.py`. Scenario = LFUD `proposition` (67 unique).
Scenario split 45:5:17 -> train/val/test (seed 42), mapped to 804 rows.

## split_scenarios.json
`{ "train": [int...], "val": [int...], "test": [int...] }`
Scenario ids are indices into the de-duplicated proposition list (stable order of
first appearance in LFUD.csv).

## valid_conclusions.jsonl  (cache, resumable, one obj per row)
`{ row_index:int, scenario_id:int, premise:str, invalid_conclusion:str,
   valid_conclusion:str }`
`valid_conclusion` may be "" if generated with --no-api or if generation failed.

## conclusion_gen_{train,val,test}.jsonl
`{ scenario_id:int, premise:str, valid_conclusion:str, invalid_conclusion:str }`
One object per LFUD row in that split. `premise` is the proposition;
`invalid_conclusion` is the original fallacious `sentence`.

## fallacy_id_{val,test}.jsonl   (from LFUD task2, 4-option MCQ)
`{ scenario_id:int, premise:str, options:[str,str,str,str], answer_idx:int }`
`premise` is task2.question, `options` is the 4 choices, `answer_idx` is the
0-based index of the correct (fallacy-containing) option per LFUD task2.answer.

The eval/baselines agent reads `conclusion_gen_*` (for generation) and
`fallacy_id_*` (for identification ΔProb/Accuracy). Schemas above are stable.
