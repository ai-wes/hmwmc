# pnn.py
from __future__ import annotations
from dataclasses import dataclass
from enum import Enum
from collections import deque
from typing import Optional, List
import numpy as np

class PNN_STATE(Enum):
    OPEN = "OPEN"
    CLOSING = "CLOSING"
    LOCKED = "LOCKED"

@dataclass(frozen=True)
class PNNConfig:
    lock_stability_window: int = 5
    lock_rel_change: float = 0.05
    closing_generations: int = 5
    min_plasticity: float = 0.2

    # budget decay
    time_decay_per_gen: float = 0.2
    stagnation_penalty_threshold: int = 6
    stagnation_penalty_multiplier: float = 2.0
    improvement_eps: float = 1e-9
    drain_success_factor: float = 0.5
    drain_failure_factor: float = 1.5

class PerineuronalNet:
    """PNN with exploitation budget and explicit force_unlock API."""
    def __init__(self, cell_id: str, exploit_budget: float = 100.0, config: PNNConfig = PNNConfig()):
        self.cfg = config
        self.cell_id = cell_id
        self.state = PNN_STATE.OPEN
        self.generations_in_phase = 0
        self.plasticity_multiplier = 1.0
        self.stability_history = deque(maxlen=10)
        self.refractory_until: Optional[int] = None

        self.exploit_budget = float(exploit_budget)
        self.initial_exploit_budget = float(exploit_budget)
        self.fitness_at_lock: Optional[float] = None
        self.best_fitness_in_lock_phase: Optional[float] = None
        self.recent_exploit_evals: float = 0.0
        self.recent_exploit_improvement: float = 0.0
        self.locked_phase_fitness_history: List[float] = []
        self.locked_phase_start_gen: Optional[int] = None

    def note_exploit_outcome(self, evals_used: float, improvement: float) -> None:
        self.recent_exploit_evals = float(max(0.0, evals_used))
        self.recent_exploit_improvement = float(improvement)

    def update(self, current_fitness: float, generation: int, island_median_fitness: Optional[float] = None) -> None:
        if self.refractory_until is not None and generation < self.refractory_until:
            self.generations_in_phase = 0
            return
        elif self.refractory_until is not None and generation >= self.refractory_until:
            self.refractory_until = None

        self.generations_in_phase += 1
        self.stability_history.append(float(current_fitness))

        if self.state == PNN_STATE.OPEN:
            if len(self.stability_history) >= self.cfg.lock_stability_window:
                recent = list(self.stability_history)[-self.cfg.lock_stability_window:]
                recent_mean = float(np.mean(recent))
                rel = abs(float(current_fitness) - recent_mean) / max(1e-8, abs(recent_mean))
                if rel < self.cfg.lock_rel_change:
                    if island_median_fitness is not None:
                        if current_fitness <= island_median_fitness:
                            self.state = PNN_STATE.CLOSING
                            self.generations_in_phase = 0
                        else:
                            self.generations_in_phase = 0  # keep OPEN
                    else:
                        self.state = PNN_STATE.CLOSING
                        self.generations_in_phase = 0

        elif self.state == PNN_STATE.CLOSING:
            progress = min(1.0, self.generations_in_phase / float(self.cfg.closing_generations))
            self.plasticity_multiplier = max(self.cfg.min_plasticity, 1.0 - (1.0 - self.cfg.min_plasticity) * progress)
            if progress >= 1.0:
                self.state = PNN_STATE.LOCKED
                self.plasticity_multiplier = self.cfg.min_plasticity
                self.generations_in_phase = 0
                self.fitness_at_lock = float(current_fitness)
                self.best_fitness_in_lock_phase = float(current_fitness)
                self.exploit_budget = float(self.initial_exploit_budget)
                self.locked_phase_fitness_history = [float(current_fitness)]
                self.locked_phase_start_gen = int(generation)

        elif self.state == PNN_STATE.LOCKED:
            self.plasticity_multiplier = self.cfg.min_plasticity
            self.locked_phase_fitness_history.append(float(current_fitness))
            if len(self.locked_phase_fitness_history) > 100:
                self.locked_phase_fitness_history = self.locked_phase_fitness_history[-100:]

            if self.best_fitness_in_lock_phase is None:
                self.best_fitness_in_lock_phase = float(current_fitness)
                stagn = 0
            elif current_fitness < self.best_fitness_in_lock_phase:
                self.best_fitness_in_lock_phase = float(current_fitness)
                stagn = 0
            else:
                stagn = self.generations_in_phase

            # budget decay
            time_decay = self.cfg.time_decay_per_gen
            if stagn >= self.cfg.stagnation_penalty_threshold:
                stagnation_decay = self.cfg.stagnation_penalty_multiplier * (1.0 + (stagn / float(self.cfg.stagnation_penalty_threshold)))
            else:
                stagnation_decay = 0.2 * (stagn / float(self.cfg.stagnation_penalty_threshold))

            used = float(self.recent_exploit_evals)
            imp = float(self.recent_exploit_improvement)
            success = imp > self.cfg.improvement_eps
            usage_decay = (self.cfg.drain_success_factor if success else self.cfg.drain_failure_factor) * used

            total = time_decay + stagnation_decay + usage_decay
            if total > 0.0:
                self.exploit_budget = max(0.0, self.exploit_budget - total)

            self.recent_exploit_evals *= 0.5
            self.recent_exploit_improvement *= 0.5

    def force_unlock(self, generation: int, refractory_period: int = 5) -> None:
        self.state = PNN_STATE.OPEN
        self.plasticity_multiplier = 1.0
        self.generations_in_phase = 0
        self.stability_history.clear()
        self.fitness_at_lock = None
        self.best_fitness_in_lock_phase = None
        self.locked_phase_fitness_history = []
        self.locked_phase_start_gen = None

        base_ref = int(refractory_period)
        recent = list(self.stability_history)[-5:]
        stress_proxy = 0.0
        if recent:
            avg = float(np.mean(recent)); std = float(np.std(recent))
            if avg > 1e-8: stress_proxy = min(1.0, std / avg)
        self.refractory_until = generation + max(refractory_period, int(round(base_ref * (1.0 + 1.5 * stress_proxy))))

"""
    # In complete_teai_methods_slim.py
    # ADD this method inside the AdvancedArchipelagoEvolution class.

    def _strategic_pnn_unlocking(self, island: List[Cell], island_idx: int, generation: int):

        unlock_candidates = []
        
        # 1. Identify all locked cells that are eligible for unlocking
        for cell in island:
            if cell.pnn.state == PNN_STATE.LOCKED:
                budget_exhausted = cell.pnn.exploit_budget <= 0
                critical_stress = cell.local_stress > cfg.pnn_stress_unlock_threshold

                if budget_exhausted or critical_stress:
                    # 2. Calculate unlock priority score
                    # Higher score for worse fitness and longer time in the locked phase
                    current_fitness = cell.fitness_history[-1] if cell.fitness_history else float('inf')
                    generations_stuck = cell.pnn.generations_in_phase
                    
                    # Priority = How bad the solution is * how long it has been unproductive
                    priority = current_fitness * (1 + 0.1 * generations_stuck)
                    
                    reason = "Budget Exhausted" if budget_exhausted else "Critical Stress"
                    unlock_candidates.append((priority, cell, reason))

        if not unlock_candidates:
            return

        # 3. Sort candidates by priority (highest priority first)
        unlock_candidates.sort(key=lambda x: x[0], reverse=True)
        
        # 4. Enforce the unlock limit (Top-K), dynamically modulated by island stress
        #    High stress => lower unlocks; low stress => easier unlocks
        island_stresses = [getattr(c, 'local_stress', 0.0) for c in island]
        stress_level = float(np.mean(island_stresses)) if island_stresses else 0.0
        # Use fraction-based unlocks depending on dimensionality
        frac = float(getattr(cfg, 'unlock_fraction_static', 0.05))
        try:
            dim = int(getattr(self, 'dimension', getattr(cfg, 'dimension', 0)))
        except Exception:
            dim = 0
        if dim >= getattr(cfg, 'high_dim_threshold', 30):
            frac = float(getattr(cfg, 'unlock_fraction_high_dim', 0.08))
        base_limit = max(1, int(round(len(unlock_candidates) * frac)))
        # scale in [0.5x, 2x] across stress in [high, low]
        stress_low = float(getattr(cfg, 'unlock_stress_low', 0.2))
        stress_high = float(getattr(cfg, 'unlock_stress_high', 0.8))
        if stress_level <= stress_low:
            scale = 2.0
        elif stress_level >= stress_high:
            scale = 0.5
        else:
            t = (stress_high - stress_level) / max(1e-8, (stress_high - stress_low))
            scale = 0.5 + 1.5 * t
        limit = max(1, int(round(base_limit * scale)))
        cells_to_unlock = unlock_candidates[:limit]
        
        logger.info(f"    STRATEGIC UNLOCK (Island {island_idx}): {len(unlock_candidates)} candidates, unlocking top {len(cells_to_unlock)}.")

        for priority, cell, reason in cells_to_unlock:
            logger.info(f"        -> Unlocking cell {cell.id} (Priority: {priority:.2f}, Reason: {reason})")
            cell.pnn.force_unlock(generation) # Use the new safe unlocking method
                        # ========================================================================

                unlock_candidates = []
                
                # 1. IDENTIFY all locked cells that are eligible to unlock
                for cell in final_island_cells:
                    if cell.pnn.state == PNN_STATE.LOCKED:
                        # Check unlock conditions (budget, stress, etc.)
                        # This logic is moved from the PNN.update method
                        
                        budget_exhausted = cell.pnn.exploit_budget <= 0
                        critical_stress = cell.local_stress > cfg.pnn_stress_unlock_threshold
                        
                        if budget_exhausted or critical_stress:
                            # 2. PRIORITIZE the unlock candidates
                            # Score = how bad is the solution + how long has it been stuck
                            fitness_score = cell.fitness_history[-1] if cell.fitness_history else float('inf')
                            stagnation_score = cell.pnn.generations_in_phase
                            
                            # We want to unlock cells with HIGH fitness (bad) and HIGH stagnation
                            unlock_priority = fitness_score + stagnation_score * 0.1 # Small weight for stagnation
                            
                            unlock_candidates.append((unlock_priority, cell))

                if unlock_candidates:
                    # Sort candidates by priority (highest priority first)
                    unlock_candidates.sort(key=lambda x: x[0], reverse=True)
                    
                    # 3. ENFORCE the unlock limit
                    limit = cfg.pnn_unlock_limit_per_gen
                    cells_to_unlock = unlock_candidates[:limit]
                    
                    logger.info(f"      STRATEGIC UNLOCK (Island {island_idx}): {len(unlock_candidates)} candidates, unlocking top {len(cells_to_unlock)}.")

                    for priority, cell in cells_to_unlock:
                        logger.info(f"         Unlocking cell {cell.id} (Priority: {priority:.2f})")
                        
                        # Perform the unlock logic here
                        cell.pnn.state = PNN_STATE.OPEN
                        cell.pnn.plasticity_multiplier = 1.0
                        cell.pnn.generations_in_phase = 0
                        cell.pnn.stability_history.clear()
                        cell.pnn.locked_phase_fitness_history.clear()
                        cell.pnn.fitness_at_lock = None
                        cell.pnn.refractory_until = generation + 5 # Set refractory period
                        unlocked_this_gen += 1

                # ========================================================================
                # END: STRATEGIC PNN UNLOCKING IMPLEMENTATION
                # ========================================================================

        
        
        
"""