from training.contracts import json_write, sha
from training.evaluate import summarize


def test_aggregate_recomputes_official_margin(tmp_path):
    maps = [301, 302]
    cases = [[i,s,m,42,"checkpoint"] for i in range(2) for s in range(2) for m in "ab"]
    json_write(tmp_path/"identity.json", {"maps":maps,"cases":cases})
    for i, case in enumerate(cases):
        idx, side, model, _, _ = case
        own = 3 if model=="b" else (2 if idx==0 else 1)
        wins = [own,5-own] if side==0 else [5-own,own]
        path = tmp_path/f"case_{i:04d}"
        path.mkdir()
        journal = [{"t":t,"reset":t%101==0,"terminal":t==505,"team_wins":wins}
                   for t in range(1,506)]
        raw = path/"case.json"
        json_write(raw, {"identity":case,"trajectory":journal,"summary":{
            "seed":maps[idx],"side":side,"opponent":"v2","goal_conflicts":0,
            "complete":True,"official_wins":wins,"game_margin":2*own-5}})
        json_write(path/"sha256.json",{"case.json":sha(raw)})
    stats = summarize(tmp_path)
    assert stats["steps"] == 8*505 and stats["fits"]["b-42"]["wins"] == 4
    assert stats["comparisons"]["b-a"]["mean"] == 3
    assert stats["comparisons"]["b-a"]["map_bootstrap95"] == [2,4]
