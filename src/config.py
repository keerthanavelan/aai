from pathlib import Path
from dataclasses import dataclass, asdict, fields
import math
import numpy as np

ROOT = Path(__file__).resolve().parents[1]


@dataclass
class SearchConfig:
    population: int = 16
    iterations: int = 15
    seed: int = 1
    train_rows: int = 10000
    validation_rows: int = 5000
    trees: int = 40
    max_depth: int | None = 16
    accuracy_weight: float = 0.9
    mutation_rate: float = 0.02439
    crossover_rate: float = 0.8
    tournament_size: int = 3
    inertia: float = 0.7
    cognitive: float = 1.5
    social: float = 1.5
    velocity_limit: float = 4.0
    min_features: int = 1
    max_features: int = 41
    patience: int = 0
    target_accuracy: float | None = None
    target_fitness: float | None = None
    max_seconds: float = 0.0
    chi_k: int = 12
    final_trees: int = 100
    attack_threshold: float = 0.5
    min_attack_recall: float | None = None
    max_false_alarm_rate: float | None = None
    objective: str = 'accuracy'
    weight_mode: str = 'none'
    rf_max_features: str | float = 'sqrt'
    exclude_constant: bool = False
    seed_ranked: bool = False

    @classmethod
    def parse(cls, data=None):
        data = dict(data or {})
        allowed = {f.name for f in fields(cls)}
        if set(data) - allowed:
            raise ValueError('Unknown config fields: ' + ', '.join(sorted(set(data)-allowed)))
        cfg = cls(**data)
        limits = {'population': (4, 100), 'iterations': (1, 100), 'seed': (0, 2**31-1),
                  'train_rows': (200, 100778), 'validation_rows': (100, 25195),
                  'trees': (5, 200), 'max_depth': (2, 40), 'tournament_size': (2, 100),
                  'min_features': (1, 41), 'max_features': (1, 41), 'patience': (0, 100),
                  'chi_k': (1, 41), 'final_trees': (5, 300)}
        for name, (low, high) in limits.items():
            value = getattr(cfg, name)
            if name=='max_depth' and value is None:continue
            if type(value) is not int or not low <= value <= high:
                raise ValueError(f'{name} must be an integer from {low} to {high}.')
        for name in ['accuracy_weight', 'mutation_rate', 'crossover_rate', 'inertia',
                     'cognitive', 'social', 'velocity_limit', 'max_seconds',
                     'target_accuracy', 'target_fitness', 'attack_threshold',
                     'min_attack_recall', 'max_false_alarm_rate']:
            value = getattr(cfg, name)
            if value is None and (name.startswith('target_') or name in ['min_attack_recall','max_false_alarm_rate']):
                continue
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
                raise ValueError(f'{name} must be a finite number.')
        for name in ['accuracy_weight', 'mutation_rate', 'crossover_rate', 'inertia']:
            if not 0 <= getattr(cfg, name) <= 1:
                raise ValueError(f'{name} must be between 0 and 1.')
        for name in ['cognitive', 'social']:
            if not 0 <= getattr(cfg, name) <= 4:
                raise ValueError(f'{name} must be between 0 and 4.')
        if not 0 < cfg.velocity_limit <= 10 or not 0 <= cfg.max_seconds <= 86400:
            raise ValueError('velocity_limit must be in (0, 10]; max_seconds in [0, 86400].')
        for name in ['target_accuracy', 'target_fitness','min_attack_recall','max_false_alarm_rate']:
            value = getattr(cfg, name)
            if value is not None and not 0 <= value <= 1:
                raise ValueError(f'{name} must be between 0 and 1, or null.')
        if not 0.05<=cfg.attack_threshold<=0.95:
            raise ValueError('attack_threshold must be between 0.05 and 0.95.')
        if cfg.min_features > cfg.max_features:
            raise ValueError('min_features must not exceed max_features.')
        if cfg.tournament_size > cfg.population:
            raise ValueError('tournament_size must not exceed population.')
        if cfg.objective not in ['accuracy','attack_aware']:
            raise ValueError('objective must be accuracy or attack_aware.')
        if cfg.weight_mode not in ['none','attack_type_sqrt','attack_type_equal']:
            raise ValueError('weight_mode must be none, attack_type_sqrt or attack_type_equal.')
        if cfg.rf_max_features!='sqrt' and (isinstance(cfg.rf_max_features,bool) or
            not isinstance(cfg.rf_max_features,(int,float)) or not 0<cfg.rf_max_features<=1):
            raise ValueError('rf_max_features must be sqrt or a fraction in (0,1].')
        if type(cfg.exclude_constant) is not bool or type(cfg.seed_ranked) is not bool:
            raise ValueError('exclude_constant and seed_ranked must be booleans.')
        return cfg

    def to_dict(self):
        return asdict(self)


def parse_mask(value, cfg=None):
    if isinstance(value, str):
        if len(value) != 41 or set(value)-{'0','1'}:
            raise ValueError('Chromosome must be a 41-character binary string.')
        mask = np.array([int(c) for c in value], dtype=np.int8)
    else:
        mask = np.asarray(value)
        if mask.shape != (41,) or not np.isin(mask, [0,1]).all():
            raise ValueError('Chromosome must be a list of exactly 41 bits.')
        mask = mask.astype(np.int8)
    low, high = (cfg.min_features, cfg.max_features) if cfg else (1,41)
    if not low <= int(mask.sum()) <= high:
        raise ValueError(f'Chromosome must select {low} to {high} features.')
    return mask


def repair(mask, rng, cfg, allowed=None):
    mask = mask.copy()
    allowed=np.ones(41,dtype=bool) if allowed is None else np.asarray(allowed,dtype=bool)
    mask[~allowed]=0
    if cfg.min_features>int(allowed.sum()):raise ValueError('min_features exceeds nonconstant feature count.')
    count = int(mask.sum())
    if count < cfg.min_features:
        mask[rng.choice(np.flatnonzero((mask == 0)&allowed), cfg.min_features-count, replace=False)] = 1
    if count > cfg.max_features:
        mask[rng.choice(np.flatnonzero(mask == 1), count-cfg.max_features, replace=False)] = 0
    return mask


def initial_positions(cfg, rng, supplied=None, allowed=None):
    supplied = supplied if supplied is not None else []
    if not isinstance(supplied, list) or len(supplied) > cfg.population:
        raise ValueError('initial_masks must be a list no longer than population.')
    positions = [repair(parse_mask(mask, cfg),rng,cfg,allowed) for mask in supplied]
    while len(positions) < cfg.population:
        positions.append(repair(rng.integers(0,2,41,dtype=np.int8),rng,cfg,allowed))
    return np.array(positions)
