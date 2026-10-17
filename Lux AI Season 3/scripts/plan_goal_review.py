"""Write preregistered one-factor configs; never starts an experiment."""
import json
from pathlib import Path

import yaml

ROOT=Path(__file__).resolve().parents[1]


def main():
    configs={
        'E1_clean':{'extends':'common.yaml'},
        'E1_dirty_control':{'extends':'E1_clean.yaml','study':{'label_view':'dirty'}},
        'E2_weighted':{'extends':'E1_clean.yaml','study':{'loss':'class_weighted'}},
        'E2_balanced':{'extends':'E1_clean.yaml','study':{'loss':'balanced_sampler'}},
        'E3_res6':{'extends':'E1_clean.yaml','model':{'architecture':'res6'}},
        'E3_res8':{'extends':'E1_clean.yaml','model':{'architecture':'res8'}},
        'E3_dilated':{'extends':'E1_clean.yaml','model':{'architecture':'dilated6'}},
        'E4_without_r':{'extends':'E1_clean.yaml','study':{'scope':'review_epochs','previous_context_dropout':0.0},'model':{'legacy_previous_bias':False}},
        'E4_with_r':{'extends':'E4_without_r.yaml','model':{'replan':True}},
        'E5_corrupted':{'extends':'E4_with_r.yaml','study':{'context':'corrupted','corruption_probability':0.5}},
        'E5_hysteresis':{'extends':'E4_with_r.yaml','model':{'hysteresis':0.25}},
        'E5_combination':{'extends':'E5_corrupted.yaml','model':{'hysteresis':0.25}},
    }
    folder=ROOT/'configs/goal_review'
    for name,cfg in configs.items():
        target=folder/(name+'.yaml')
        text=yaml.safe_dump(cfg,sort_keys=False)
        if target.exists() and target.read_text()!=text:raise ValueError('config already differs; preserve preregistration')
        target.write_text(text)
    print(json.dumps({'configs':list(configs),'training_started':False}))


if __name__=='__main__':main()
