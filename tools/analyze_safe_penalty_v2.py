"""Diagnose penalty gradients on real saved v2 trajectories, without training.

A gradient/update alignment is conditional on one sampled batch. It does not
establish out-of-sample collision improvement or convergence of the constraint.
Gradient clipping does not imply a proportionally smaller Adam parameter step;
this analysis cannot establish the cause of the previous v1 result.
No simulator rollout, checkpoint mutation, or runtime source edit is performed.
"""
from __future__ import annotations

import argparse
from collections import Counter
import csv
import json
from pathlib import Path

import numpy as np
import torch

from gp8_control.rl.safe_v2.ppo import SafePPO, calculate_gae
from gp8_control.tools.train_safe_simulator_v2 import ROOT, sha, source_hashes, save_json


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def gradients(loss, parameters):
    result=torch.autograd.grad(loss,parameters,allow_unused=True,retain_graph=True)
    return torch.cat([(torch.zeros_like(p) if g is None else g).reshape(-1)
                      for p,g in zip(parameters,result)]).detach()


def norm(vector):
    return float(torch.linalg.vector_norm(vector.double()))


def alignment(update, gradient):
    descent=-gradient
    dot=float(torch.dot(update.double(),descent.double()))
    denominator=norm(update)*norm(descent)
    return dict(dot=dot,cosine=(dot/denominator if denominator>1e-15 else None))


def read_batch(path):
    with np.load(path) as data:
        arrays={key:data[key].copy() for key in data.files}
    rows=[{key:arr[i].item() if arr[i].ndim==0 else arr[i] for key,arr in arrays.items()}
          for i in range(len(arrays['head']))]
    return arrays,rows


def analyze_batch(out, checkpoint, label, update, protocol):
    previous_path=checkpoint/f'safe_seed{label}_update{update-1:03d}.pt'
    current_path=checkpoint/f'safe_seed{label}_update{update:03d}.pt'
    policy=SafePPO(protocol['model_initialization_seeds'][label],obs_size=protocol['observation_size'])
    metadata=policy.load(previous_path)
    after=SafePPO(protocol['model_initialization_seeds'][label],obs_size=protocol['observation_size'])
    after_metadata=after.load(current_path)
    require(metadata['update']==update-1 and after_metadata['update']==update,'Checkpoint boundaries do not match the saved batch')
    require(metadata['protocol_sha256']==sha(out/'protocol.json')==after_metadata['protocol_sha256'],
            'Checkpoint frozen protocol differs')
    batch_path=out/'batches'/f'seed{label}_update{update:03d}.npz'
    arrays,rows=read_batch(batch_path)
    gae=calculate_gae(rows)
    obs=torch.as_tensor(arrays['obs'],dtype=torch.float32)
    heads=torch.as_tensor(arrays['head'],dtype=torch.long)
    latent=torch.as_tensor(arrays['latent'],dtype=torch.float32)
    old_logp=torch.as_tensor(arrays['logp'],dtype=torch.float32)
    np.testing.assert_allclose(policy.scaler.normalize(arrays['raw_obs']),arrays['obs'],rtol=0,atol=1e-6)
    distribution,values=policy.network(obs,heads)
    actions=torch.tanh(latent)
    logp=(distribution.log_prob(latent)-torch.log(1.-actions.square()+1e-6)).sum(-1)
    ratio=(logp-old_logp).exp()
    logp_error=float((logp-old_logp).abs().max().detach())
    value_error=float(np.max(np.abs(values.detach().numpy()-arrays['value'])))
    require(max(logp_error,value_error)<1e-4,'Batch was not collected by the declared behavior checkpoint')
    actor_parameters=list(policy.network.actor.parameters())
    critic_parameters=list(policy.network.reward_critic.parameters())+list(policy.network.cost_critic.parameters())
    require(not {id(p) for p in actor_parameters}&{id(p) for p in critic_parameters},'Actor and critics share parameters')
    advantages=gae['advantages']
    multiplier=policy.multiplier
    combined=advantages[:,0]-multiplier*advantages[:,1]
    scale=float(combined.std()+1e-8)
    normalized=torch.as_tensor((combined-combined.mean())/scale,dtype=torch.float32)
    ar=torch.as_tensor(advantages[:,0]-advantages[:,0].mean(),dtype=torch.float32)
    ac=torch.as_tensor(advantages[:,1]-advantages[:,1].mean(),dtype=torch.float32)
    cost_objective=(ratio*ac).mean()
    reward_objective=(ratio*ar).mean()
    policy_loss=torch.maximum(-normalized*ratio,-normalized*ratio.clamp(.8,1.2)).mean()
    entropy=distribution.entropy().sum(-1).mean()
    actor_loss=policy_loss-.001*entropy
    g_cost=gradients(cost_objective,actor_parameters)
    g_reward=gradients(reward_objective,actor_parameters)
    g_combined=gradients(policy_loss,actor_parameters)
    g_actor=gradients(actor_loss,actor_parameters)
    error=values-torch.from_numpy(gae['targets'])
    raw_critic_loss=error.square().mean()
    normalized_critic_loss=(error/torch.as_tensor(policy.return_scales,dtype=torch.float32)).square().mean()
    g_critic_raw=gradients(raw_critic_loss,critic_parameters)
    g_critic=gradients(normalized_critic_loss,critic_parameters)
    cross_critic_actor=gradients(normalized_critic_loss,actor_parameters)
    cross_actor_critic=gradients(actor_loss,critic_parameters)
    require(norm(cross_critic_actor)==norm(cross_actor_critic)==0.,'Critic and actor gradients are not isolated')
    before_vector=torch.cat([p.detach().reshape(-1) for p in actor_parameters])
    after_vector=torch.cat([p.detach().reshape(-1) for p in after.network.actor.parameters()])
    change=after_vector-before_vector
    stats_path=out/'batch_stats'/f'update{update:03d}_seed{label}.json'
    stats=json.loads(stats_path.read_text())
    require(abs(stats['multiplier_before']-multiplier)<1e-10,'Lagrange multiplier differs from behavior checkpoint')
    contacts=np.asarray(arrays.get('contacts',np.zeros(len(rows))),float)
    # This auxiliary direction uses real solver contact counts with a zero
    # contact-specific baseline. It is explicitly distinct from trained A_C.
    contact_rows=[dict(row,reward=0.,cost=float(contacts[i]),value=np.zeros(2)) for i,row in enumerate(rows)]
    contact_adv=calculate_gae(contact_rows)['advantages'][:,1]
    contact_centered=torch.as_tensor(contact_adv-contact_adv.mean(),dtype=torch.float32)
    g_contact=gradients((ratio*contact_centered).mean(),actor_parameters)
    roles=np.asarray(arrays.get('role',arrays.get('phase',np.full(len(rows),'unknown')))).astype(str)
    by_role={}
    for role in dict.fromkeys(roles):
        mask=torch.as_tensor(roles==role)
        role_gradient=gradients((ratio[mask]*ac[mask]).mean(),actor_parameters)
        by_role[str(role)]=dict(decisions=int(mask.sum()),contacts=float(contacts[roles==role].sum()),
            cost_advantage_std=float(advantages[roles==role,1].std()),cost_gradient_norm=norm(role_gradient))
    raw_mse=error.square().mean(dim=0).detach().numpy()
    return dict(model=label,initialization_seed=protocol['model_initialization_seeds'][label],update=update,
        decisions=len(rows),actual_contact_events=float(contacts.sum()),contact_rows=int((contacts>0).sum()),
        positive_cost_rows=int((arrays['cost']>0).sum()),lambda_before=multiplier,
        reward_advantage_mean=float(advantages[:,0].mean()),reward_advantage_std=float(advantages[:,0].std()),
        cost_advantage_mean=float(advantages[:,1].mean()),cost_advantage_std=float(advantages[:,1].std()),
        combined_advantage_std=float(combined.std()),frozen_return_scales=policy.return_scales.tolist(),
        reward_critic_raw_mse=float(raw_mse[0]),cost_critic_raw_mse=float(raw_mse[1]),
        raw_critic_loss=float(raw_critic_loss.detach()),normalized_critic_loss=float(normalized_critic_loss.detach()),
        cost_only_actor_gradient_norm=norm(g_cost),reward_only_actor_gradient_norm=norm(g_reward),
        weighted_cost_contribution_norm=norm(multiplier*g_cost/scale),
        combined_policy_actor_gradient_norm=norm(g_combined),actual_actor_loss_gradient_norm=norm(g_actor),
        expected_actor_gradient_norm_after_clip=min(norm(g_actor),.5),
        raw_critic_gradient_norm=norm(g_critic_raw),normalized_critic_gradient_norm=norm(g_critic),
        critic_gradient_on_actor_norm=norm(cross_critic_actor),actor_gradient_on_critics_norm=norm(cross_actor_critic),
        decomposition_relative_error=norm(g_combined-(-g_reward+multiplier*g_cost)/scale)/max(norm(g_combined),1e-12),
        stored_actual_actor_parameter_change_norm=norm(change),
        stored_update_vs_cost_descent=alignment(change,g_cost),
        stored_update_vs_combined_descent=alignment(change,g_combined),
        solver_contact_only_zero_baseline_gradient_norm=norm(g_contact),
        stored_update_vs_contact_descent_auxiliary=alignment(change,g_contact),
        cost_gradient_nonzero=norm(g_cost)>1e-10,critic_cannot_clip_actor=True,
        actual_training_minibatch_mean_actor_gradient_norm=stats['actor_grad_norm'],
        actual_training_minibatch_mean_critic_gradient_norm=stats['critic_grad_norm'],
        workload_cost_per_offered=stats['cost_per_offered'],budget_per_offered=stats['cost_budget_per_offered'],
        by_role=by_role,behavior_logp_max_error=logp_error,behavior_value_max_error=value_error,
        artifacts_sha256=dict(behavior_checkpoint=sha(previous_path),after_checkpoint=sha(current_path),batch=sha(batch_path)))


def evaluation_closure(out, protocol):
    files=sorted((out/'evaluation').glob('*.json'))
    if not files:
        return dict(available=False,complete=False,rollouts=0)
    cost=protocol['cost'];reward=protocol['reward']
    checks=[]
    contacts_in_rows=0
    for path in files:
        result=json.loads(path.read_text())
        summary,trace,rows=result['summary'],result['trace'],result['decisions']
        active=[r for r in trace['residuals'] if r.get('active',True)]
        baseline=result['policy']=='baseline'
        require(baseline or len(active)==len(rows),'Physical residual records do not match neural decisions')
        max_reward_error=max_cost_error=max_contact_error=0.
        contact_rows=0
        for index,row in enumerate(rows):
            components=row['reward_components']
            count=float(row['contacts']);impulse=float(row['impulse_ns']);falls=float(row['collateral_falls'])
            rejected=float(active[index].get('rejected',False))
            physical_failure=float(bool(row['terminated']) and summary['termination_reason']=='physics_nonfinite')
            expected_cost=cost['contact_event']*count+cost['impulse_ns']*impulse+cost['collateral_fall']*falls
            expected_cost+=cost['rejected_residual']*rejected+cost['physics_nonfinite']*physical_failure
            reward_error=abs(float(row['reward'])-sum(components.values()))
            cost_error=abs(float(row['cost'])-expected_cost)
            contact_error=abs(float(components['contact'])-reward['contact_event']*count)
            max_reward_error=max(max_reward_error,reward_error)
            max_cost_error=max(max_cost_error,cost_error)
            max_contact_error=max(max_contact_error,contact_error)
            if count>0:
                require(row['cost']>0 and components['contact']<0,'Real contact did not produce cost/negative contact reward')
                contact_rows+=1;contacts_in_rows+=int(count)
        row_cost=sum(float(r['cost']) for r in rows)
        assigned_contacts=sum(int(r['contacts']) for r in rows)
        require(abs(row_cost-summary['transition_assigned_cost'])<1e-7,'Assigned transition cost differs')
        require(abs(row_cost+summary['initial_unassigned_cost']-summary['total_cost'])<1e-7,'Full episode cost was lost before the first decision')
        require(assigned_contacts+summary['initial_unassigned_contact_events']==summary['contacts'],'Full episode contact assignment does not close')
        require(max(max_reward_error,max_cost_error,max_contact_error)<1e-7,'Actual penalty arithmetic differs from protocol')
        checks.append(dict(policy=result['policy'],scene_seed=result['scene_seed'],decisions=len(rows),
            contacts=summary['contacts'],contact_rows=contact_rows,assigned_contacts=assigned_contacts,
            initial_unassigned_contacts=summary['initial_unassigned_contact_events'],
            max_reward_error=max_reward_error,max_cost_error=max_cost_error,max_contact_reward_error=max_contact_error))
    completion=out/'evaluation_summary.json'
    complete=completion.is_file() and bool(json.loads(completion.read_text())['complete'])
    return dict(available=True,complete=complete,rollouts=len(checks),real_contact_events_assigned_to_rows=contacts_in_rows,
                all_real_contacts_penalized=True,checks=checks)


def analyze(run, updates=None):
    out=ROOT/'runs'/run;checkpoint=ROOT/'checkpoints'/run
    protocol=json.loads((out/'protocol.json').read_text())
    require(protocol['source_sha256']==source_hashes(),'Frozen runtime/trainer sources changed')
    committed=int(json.loads((out/'committed_update.json').read_text())['update'])
    if updates is None:
        updates=sorted({1,committed})
    require(all(1<=update<=committed for update in updates),'Requested update has not been fully committed')
    checks=[analyze_batch(out,checkpoint,label,update,protocol) for update in updates for label in (0,1)]
    result=dict(run=run,committed_update=committed,planned_updates=protocol['training']['updates'],
        final_training_complete=committed==protocol['training']['updates'],analyzed_updates=updates,
        gradient_checks=checks,evaluation_penalty_closure=evaluation_closure(out,protocol),
        interpretation=['Gradients are full saved-batch gradients at the behavior checkpoint; logged training norms average randomized minibatches over several epochs.',
            'A nonzero sampled cost gradient shows a learnable penalty signal on these trajectories, not out-of-sample collision improvement.',
            'The stored actor update includes reward, cost, entropy, clipping and multiple Adam steps; its cost-descent cosine can be negative.',
            'Contact-only gradient is an auxiliary zero-baseline GAE of actual solver contact counts; it is not the trained cost-critic advantage.',
            'Independent actor/critic parameter graphs and separate clipping prevent critic gradients from entering the actor clipping norm.',
            'Rescaling raw gradients does not imply proportional Adam parameter-step reduction: Adam can cancel a constant uniform scale.',
            'Gradient isolation is a v2 implementation property; it does not establish that shared gradient clipping caused the previous v1 evaluation result.'],
        protocol_sha256=sha(out/'protocol.json'))
    save_json(out/'penalty_analysis_v2.json',result)
    fields=['model','update','actual_contact_events','contact_rows','cost_advantage_std','lambda_before',
        'cost_only_actor_gradient_norm','combined_policy_actor_gradient_norm','normalized_critic_gradient_norm',
        'critic_gradient_on_actor_norm','stored_actual_actor_parameter_change_norm','cost_descent_dot','cost_descent_cosine']
    with (out/'penalty_analysis_v2.csv').open('w',newline='') as handle:
        writer=csv.DictWriter(handle,fieldnames=fields);writer.writeheader()
        for item in checks:
            row={key:item[key] for key in fields if key in item}
            row.update(cost_descent_dot=item['stored_update_vs_cost_descent']['dot'],
                       cost_descent_cosine=item['stored_update_vs_cost_descent']['cosine'])
            writer.writerow(row)
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',default='feature_safe_route_20261021_v2')
    parser.add_argument('--updates',help='Comma-separated committed update numbers; default first and latest committed')
    args=parser.parse_args()
    result=analyze(args.run,None if args.updates is None else [int(value) for value in args.updates.split(',')])
    print(json.dumps(dict(run=args.run,updates=result['analyzed_updates'],
        nonzero_cost_gradients=sum(c['cost_gradient_nonzero'] for c in result['gradient_checks']),
        gradient_checks=len(result['gradient_checks']),evaluation_complete=result['evaluation_penalty_closure']['complete']),ensure_ascii=False))


if __name__=='__main__':main()
