import numpy as np
from scripts.fusion_validation.feature_data import drop_type_matrix,geometry_columns
from scripts.fusion_validation.fusion_metrics import graph_metrics,paired_summary,clustered_paired_summary
import pandas as pd


def test_recall_denominator_is_all_positives():
    y=np.r_[np.ones(20),np.zeros(10)]
    score=np.arange(30,0,-1)
    metrics=graph_metrics(y,score)
    assert metrics['recall_at_10']==.5
    assert np.isclose(metrics['graph_AP'],1)
    assert metrics['IoU_at_P']==1


def test_empty_positive_graph_is_explicitly_undefined():
    assert np.isnan(graph_metrics(np.zeros(4),np.arange(4))['graph_AP'])


def test_paired_ci_preserves_complex_pairing():
    rows=[dict(cohort='test',sample_id=str(i),method=method,graph_AP=value,
               IoU_at_P=value,recall_at_10=value)
          for i in range(5) for method,value in [('LAIGN',.9),('Control',.8)]]
    result=paired_summary(pd.DataFrame(rows),bootstrap=100)
    assert np.allclose(result.delta,.1)
    assert np.allclose(result.ci95_low,.1)
    assert np.allclose(result.ci95_high,.1)


def test_type_removal_recomputes_shared_scalar_not_only_named_columns():
    names=['visnet','visnet_logit','visnet_over_1_plus_distance']
    names += [f'visnet_x_range_tau_{tau}' for tau in (1,2,3,4,6,8,10,12,16,20)]
    names += ['typed_multiscale__any_interaction_pair_max','typed_multiscale__any_interaction_pair_mean',
              'typed_multiscale__any_interaction_top3_mean']
    names += [f'typed_multiscale__any_interaction_weighted_tau_{tau}' for tau in (4,8)]
    names += [f'typed_multiscale__any_interaction_{stat}_ge_{t:g}' for t in (.3,.5,.7) for stat in ('count','frac')]
    names += ['typed_multiscale__salt_bridge_'+stat for stat in ('max','mean','weighted_tau_4')]
    names += ['other_'+str(i) for i in range(73-len(names))]
    probs=np.full((6,5),.2,dtype=np.float32)
    probs[:,2]=.99
    z={'feature_names':np.asarray(names),'interaction_names':np.asarray(['hydrophobic_contact','hydrogen_bond','salt_bridge','pi_stacking','pi_cation']),
       'y':np.ones(2),'residue_for_pair':np.repeat(np.arange(2),3),'pair_prob':probs,
       'pair_distance':np.full((6,2),4.),'distance':np.array([4.,4.]),'x':np.full((2,73),.99,dtype=np.float32)}
    dropped=drop_type_matrix(z,'salt_bridge')
    assert dropped.shape==(2,70)
    assert np.allclose(dropped[:,0],.2)
    assert np.allclose(dropped[:,names.index('typed_multiscale__any_interaction_pair_max')],.2)


def test_receptor_grouped_bootstrap_keeps_entire_clusters():
    frame=pd.DataFrame([dict(cohort='temporal',sample_id=str(i),method=m,graph_AP=v,
        IoU_at_P=v,recall_at_10=v) for i in range(6) for m,v in [('LAIGN',.9),('Control',.8)]])
    groups=pd.Series({str(i):str(i//2) for i in range(6)})
    out=clustered_paired_summary(frame,groups,bootstrap=100)
    assert (out.receptor_sequence_groups==3).all()
    assert (out.complexes==6).all()
    assert np.allclose(out.ci95_low,.1) and np.allclose(out.ci95_high,.1)
