# src/morota/sim/agent/pedestrian_agent.py
from __future__ import annotations

from collections import Counter
from typing import Dict, List, Mapping, TYPE_CHECKING, Literal, Optional, Tuple
import math

from mesa import Agent, Model


class StationAgent(Agent):
    """
    StationAgent
    """

    def __init__(
        self,
        model: Model,
    ):
        super().__init__(model)


    def step(self):
        # 信号を見る
        # 移動する or 待つ
        # ゴール判定
        pass
