"""Offline fail-closed tests for pairing, public replay and cluster statistics."""
import copy
import io
import json

import numpy as np
import pytest
import torch

from research.evidence import read_evidence
from research.paired_goal_credit.public_replay import Journal, RecordedPublic, official_margin
from research.paired_goal_credit.report import reconstruct
from research.paired_goal_credit.state import digest
from research.paired_goal_credit.statistics import summarize_rows
from research.paired_goal_credit.streams import continuation_draw, fixed_pair, verify_fixed_node

def test_pair_duplicates_retained_and_rng_not_consumed():
    state=torch.get_rng_state().clone()
    p=torch.tensor([1.,0.,0.]);mask=torch.tensor([True,True,False])
    assert fixed_pair(p,mask,"example",32,4)==[0,0]
    assert torch.equal(state,torch.get_rng_state())

def test_illegal_sampling_mass_rejected():
    with pytest.raises(RuntimeError):
        fixed_pair(torch.tensor([.4,.4,.2]),[True,True,False],"x",32,1)

def test_decision_identity_prevents_call_count_coupling():
    a=continuation_draw("x",0,40,3,"Goal")
    for t in range(41,100):continuation_draw("x",0,t,7,"Mode")
    assert a==continuation_draw("x",0,40,3,"Goal")
    assert a!=continuation_draw("x",1,40,3,"Goal")

def test_fixed_state_certificate_rejects_support_or_pair_changes():
    node=read_evidence("PGC01.json")["rows"][0]["node"]
    verify_fixed_node(node,copy.deepcopy(node))
    for field,new in [("pair",[0,1]),("legal",[]),("mode",-1),("earlier",[])]:
        changed=copy.deepcopy(node);changed[field]=new
        with pytest.raises(RuntimeError):verify_fixed_node(node,changed)

def test_state_digest_preserves_type_and_mutation():
    value={"u":np.array([1,2]),"goal":(2,3)}
    assert digest(value)==digest(copy.deepcopy(value))
    other=copy.deepcopy(value);other["u"][0]=3
    assert digest(value)!=digest(other)
    with pytest.raises(TypeError):digest(np.array([object()],dtype=object))
    cyclic=[];cyclic.append(cyclic)
    with pytest.raises(RuntimeError):digest(cyclic)

def test_public_prefix_requires_exact_records_and_exhausted_tail():
    line=json.dumps({"kind":"goal","uid":1},separators=(",",":"))+"\n"
    journal=Journal(io.StringIO(),io.StringIO(line))
    with pytest.raises(RuntimeError):journal.fork()
    journal=Journal(io.StringIO(),io.StringIO(line))
    with pytest.raises(RuntimeError):journal.put({"kind":"goal","uid":2})
    journal=Journal(io.StringIO(),io.StringIO(line))
    journal.put({"kind":"goal","uid":1});journal.fork()

def test_public_tape_rejects_counterfactual_actions_and_no_shared_mutation():
    trace=[{"kind":"public_ipc","request":{"step":0,"action":[0]},"response":{"obs":{"x":[1]},"_pgc_audit":{"hidden_hash":"opaque"}}}]
    tape=RecordedPublic(trace);response=tape.request(step=0,action=[0])
    assert "_pgc_audit" not in response
    response["obs"]["x"][0]=9
    assert trace[0]["response"]["obs"]["x"][0]==1
    tape.finish()
    with pytest.raises(RuntimeError):RecordedPublic(trace).request(step=0,action=[1])
    with pytest.raises(RuntimeError):RecordedPublic(trace).finish()

@pytest.mark.parametrize("side",[0,1])
def test_terminal_official_margin_direction(side):
    wins=[3,2];oriented=wins if side==0 else wins[::-1]
    response={"task_complete":True,"obs":{"team_wins":wins},"official_reward":oriented}
    assert official_margin(response,side)==(1 if side==0 else -1)
    response["official_reward"]=[0,0]
    with pytest.raises(RuntimeError):official_margin(response,side)

def test_historical_rows_rebuild_maps_not_pseudoreplicates():
    result=reconstruct()
    assert result["maps"]==24 and result["observed_contexts"]==48
    assert result["repeated_pairs"]==96 and result["zero_deltas"]==96
    assert result["distinct_goal_contexts"]==37 and result["action_divergences"]==53
    assert result["95pct_map_interval"]==[0.,0.] and result["bootstrap_draws"]==20000

def test_duplicate_omitted_and_fabricated_missing_rows_rejected():
    saved=read_evidence("PGC01.json");contexts=read_evidence("PGC01_manifest.json")["contexts"]
    stats=read_evidence("PGC01_protocol.json")["statistic"]
    with pytest.raises(RuntimeError):summarize_rows(saved["rows"][:-1],contexts,stats)
    rows=copy.deepcopy(saved["rows"]);rows[0]["missing"]=True
    with pytest.raises(RuntimeError):summarize_rows(rows,contexts,stats)
    rows=copy.deepcopy(saved["rows"]);rows[-1]=copy.deepcopy(rows[0])
    with pytest.raises(RuntimeError):summarize_rows(rows,contexts,stats)

def test_all_historical_goal_pairs_reproduce_from_original_probabilities():
    for row in read_evidence("PGC01.json")["rows"]:
        if row["missing"]:
            continue
        node=row["node"]
        legal=torch.zeros(576,dtype=torch.bool)
        legal[node["legal"]]=True
        pair=fixed_pair(torch.tensor(node["goal_probs"],dtype=torch.float64),legal,
                        row["context"],node["step"],node["uid"])
        assert pair==node["pair"]

@pytest.mark.parametrize("deltas,product,status",[
    ([1,2],1,"SIGNAL_SUPPORTED"),
    ([1,-1],-1,"STOP_SIGNAL_NOT_ESTABLISHED"),
])
def test_sign_statistic_nonzero_controls(deltas,product,status):
    rows=copy.deepcopy(read_evidence("PGC01.json")["rows"])
    contexts=read_evidence("PGC01_manifest.json")["contexts"]
    stats=read_evidence("PGC01_protocol.json")["statistic"]
    for row in rows:
        row["deltas"]=deltas
        row["product"]=product
    result=summarize_rows(rows,contexts,stats)
    assert result["status"]==status
    assert result["95pct_map_interval"]==[product,product]

def test_missing_context_stays_in_registered_denominator():
    rows=copy.deepcopy(read_evidence("PGC01.json")["rows"])
    contexts=read_evidence("PGC01_manifest.json")["contexts"]
    stats=read_evidence("PGC01_protocol.json")["statistic"]
    for row in rows:
        row["deltas"]=[1,1]
        row["product"]=1
    rows[0]["missing"]=True
    rows[0]["deltas"]=None
    result=summarize_rows(rows,contexts,stats)
    assert result["missing_contexts"]==1 and result["observed_contexts"]==47
    assert result["same_sign_fraction_all"]==47/48
