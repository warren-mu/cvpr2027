"""Approved supplement-4 adapter, leaving queued historical producers intact."""
import copy

from . import pair_v2_shard as base
from . import pair_v2_stable_shard as stable
from . import pair_v2_stable_policy as prior_policy
from . import pair_v2_search_shard as search
from . import pair_v2_solvable_policy as policy
from . import pair_v2_role_policy as roles

_SEARCH_ENABLE = search.enable_config
_SEARCH_GENERATOR = search.generator_worker
_ROLE_DOMAINS = None


def enable_config(config):
    _SEARCH_ENABLE(config)
    config.update(solvability_version=policy.SOLVABILITY_VERSION,
                  solvability_policy=copy.deepcopy(policy.SOLVABILITY_POLICY),
                  generation_version=roles.ROLE_VERSION, generation_policy=copy.deepcopy(roles.ROLE_POLICY),
                  validation_scope='T01v2 construction and supplement4(a) one-step observability; T02 P-table/BFS checks pending',
                  producer_module='src.envs.pair_v2_solvable_shard')
    return config


def export_pair(original, config, row, worlds, prefix, after, witnesses, constructions):
    from .pair_v2_solvable_export import export_pair as export_with_observability
    try:
        return export_with_observability(original, config, row, worlds, prefix, after, witnesses, constructions)
    except ValueError as error:
        if str(error) == 'Portable solvability proof exceeded its declared byte limit':
            raise search.EvidenceIncomplete(str(error)) from error
        raise


def proposal_iterator(sim, baseline, identity, reachable, config):
    """Frozen authenticated native filters, followed by versioned view ordering."""
    _, restored = sim.restore_snapshot(baseline)
    sim.store.save_json(f'proposal_setup_{sim.action_count:06d}.json', {
        'purpose': 'common-baseline setup before native placement query; not a w0 physics trial',
        'identity': copy.deepcopy(identity), 'baseline_reference': baseline.get('reference'),
        'restoration': copy.deepcopy(restored)})
    if restored.get('restore_deterministic') is not True:
        raise RuntimeError('Common baseline state restoration failed before native proposal queries')
    objects = search.engine.objects_by_name(baseline['objects'])
    a, b, x = (objects[name] for name in (*identity['candidate_names'], identity['anchor_name']))
    receptacles = sorted((obj for obj in objects.values() if obj.get('receptacle')
        and obj['name'] not in (*identity['candidate_names'], identity['anchor_name'])
        and not (obj.get('openable') and not obj.get('isOpen'))),
        key=lambda obj: (min(base.distance(obj['position'], candidate['position']) for candidate in (a, b)), obj['name']))
    points = []
    for obj in receptacles[:config['max_receptacles']]:
        try:
            for token in sim.spawn_points(obj['name']):
                points.append({'position': copy.deepcopy(token['selected_position']),
                               'receptacle_name': token['receptacle_name'], 'spawn': token})
        except (TimeoutError, KeyboardInterrupt, SystemExit):
            raise
        except Exception as error:
            sim.store.save_json(f'spawn_failure_{sim.action_count:06d}.json',
                                {'receptacle': obj['name'], 'error': str(error)})
    left, right = base.placement_groups(points, a, b, x, maximum_per_side=config['max_points_per_side'])
    views = base.view_proposals(reachable, a, b, maximum_positions=config['max_view_positions'])
    stream = roles.view_expanded_proposals(left, right, views)
    key = search.RoundRobinScheduler.key(identity)

    def update():
        if _ROLE_DOMAINS is not None:
            _ROLE_DOMAINS[key] = {'identity': copy.deepcopy(identity),
                'baseline_reference': copy.deepcopy(baseline.get('reference')),
                'scope': 'actual native-filtered proposal enumeration, not accepted-data balance',
                'domain': copy.deepcopy(stream.summary())}

    update()
    try:
        for proposal, view, domain in stream:
            update()
            proposal = copy.deepcopy(proposal)
            proposal['role_domain'] = copy.deepcopy(domain)
            yield proposal, view
    finally:
        update()


def pair_attempt_index(pair_id, scene):
    prefix = scene.lower() + '_'
    if not pair_id.startswith(prefix) or not pair_id[len(prefix):].isdigit():
        raise ValueError('Role assignment requires the actual numeric scene attempt identity')
    index = int(pair_id[len(prefix):])
    if pair_id != f'{scene.lower()}_{index:05d}':
        raise ValueError('Noncanonical scene attempt identity')
    return index


def generator_worker(config, connection):
    global _ROLE_DOMAINS
    if (config.get('solvability_version') != policy.SOLVABILITY_VERSION
            or config.get('solvability_policy') != policy.SOLVABILITY_POLICY
            or config.get('generation_version') != roles.ROLE_VERSION
            or config.get('generation_policy') != roles.ROLE_POLICY):
        raise ValueError('Solvable worker requires the explicit approved version and exact policy')
    previous = (base.build_pair, base.export_pair, prior_policy.choose_look,
                prior_policy.classify_pair, stable.compact_attempt, search.summary,
                search.proposal_iterator, _ROLE_DOMAINS)

    def annotate(row, report=None):
        row.update(solvability_version=policy.SOLVABILITY_VERSION,
                   solvability_policy=copy.deepcopy(policy.SOLVABILITY_POLICY),
                   generation_version=roles.ROLE_VERSION, generation_policy=copy.deepcopy(roles.ROLE_POLICY))
        if row.get('kind', '').startswith('look') and report is not None:
            row['look_solvability'] = copy.deepcopy(report)
            row['observation_search'] = copy.deepcopy(report)
            for key in ('anchor_visible_actions_w0', 'anchor_visible_actions_w1'):
                row[key] = copy.deepcopy(report.get(key))

    def build(sim, child_config, baseline, identities, proposal, view, pair_id, directory, reachable, attempt_row=None):
        row = {} if attempt_row is None else attempt_row
        policy.reset_report()
        annotate(row)
        error = None
        try:
            proposal, role = roles.orient_proposal(proposal, child_config['scene'],
                pair_attempt_index(pair_id, child_config['scene']), seed=child_config['seed'])
            names = identities['candidate_names']
            if names != sorted(names) or len(names) != 2:
                raise ValueError('Candidate stable names must retain their original sorted roles')
            row['proposal_role'] = {**role, 'candidate_a_name': names[0], 'candidate_b_name': names[1]}
            return previous[0](sim, child_config, baseline, identities, proposal, view,
                               pair_id, directory, reachable, attempt_row=row)
        except BaseException as caught:
            error = caught
            raise
        finally:
            report = policy.last_report()
            annotate(row, report)
            if error is None and row.get('kind', '').startswith('look') and report is not None:
                checked = policy.solvability_summary(report)
                if not checked['passed']:
                    row['status'] = 'rejected'
                    reasons = set(row.get('discard_reasons') or [])
                    reasons.discard('one_action_evidence_unavailable')
                    row['discard_reasons'] = sorted(reasons | set(checked['reasons']))
            policy.reset_report()

    def exporter(child_config, row, worlds, prefix, after, witnesses, constructions):
        annotate(row, row.get('observation_search') or policy.last_report())
        if row.get('kind', '').startswith('look'):
            report = row.get('look_solvability') or {}
            if not policy.solvability_summary(report)['passed']:
                raise ValueError('Look export requires both measured visibility sets and complete native evidence')
        return export_pair(previous[1], child_config, row, worlds, prefix, after, witnesses, constructions)

    def compact(row):
        result = previous[4](row)
        result.update(solvability_version=policy.SOLVABILITY_VERSION,
                      look_solvability=policy.compact_report(row.get('look_solvability')),
                      generation_version=roles.ROLE_VERSION, proposal_role=copy.deepcopy(row.get('proposal_role')))
        for key in ('anchor_visible_actions_w0', 'anchor_visible_actions_w1'):
            result[key] = copy.deepcopy(row.get(key))
        return result

    def summary(state, child_config):
        result = previous[5](state, child_config)
        result.update(solvability_version=policy.SOLVABILITY_VERSION,
                      solvability_policy=copy.deepcopy(policy.SOLVABILITY_POLICY),
                      generation_version=roles.ROLE_VERSION, generation_policy=copy.deepcopy(roles.ROLE_POLICY),
                      role_proposal_domains=copy.deepcopy(list((_ROLE_DOMAINS or {}).values())),
                      p_backend_solvability_verified=None)
        return result

    try:
        base.build_pair, base.export_pair = build, exporter
        prior_policy.choose_look, prior_policy.classify_pair = policy.choose_look, policy.classify_pair
        stable.compact_attempt, search.summary = compact, summary
        search.proposal_iterator, _ROLE_DOMAINS = proposal_iterator, {}
        result = _SEARCH_GENERATOR(config, connection)
        result.update(solvability_version=policy.SOLVABILITY_VERSION,
                      solvability_policy=copy.deepcopy(policy.SOLVABILITY_POLICY),
                      generation_version=roles.ROLE_VERSION, generation_policy=copy.deepcopy(roles.ROLE_POLICY),
                      role_proposal_domains=copy.deepcopy(list(_ROLE_DOMAINS.values())),
                      p_backend_solvability_verified=None)
        return result
    finally:
        (base.build_pair, base.export_pair, prior_policy.choose_look,
         prior_policy.classify_pair, stable.compact_attempt, search.summary,
         search.proposal_iterator, _ROLE_DOMAINS) = previous
        policy.reset_report()


def main():
    previous = search.generator_worker, search.enable_config, base.write_result

    def writer(name, metrics, config):
        enable_config(config)
        metrics.update(solvability_version=policy.SOLVABILITY_VERSION,
                       solvability_policy=copy.deepcopy(policy.SOLVABILITY_POLICY),
                       generation_version=roles.ROLE_VERSION, generation_policy=copy.deepcopy(roles.ROLE_POLICY),
                       p_backend_solvability_verified=None)
        return previous[2](name, metrics, config)

    try:
        search.generator_worker, search.enable_config, base.write_result = generator_worker, enable_config, writer
        return search.main()
    finally:
        search.generator_worker, search.enable_config, base.write_result = previous


if __name__ == '__main__':
    raise SystemExit(main())
