"""
AURA-M Fleet & Engine Graph Intelligence Engine (Technology Module 10)
Constructs NetworkX graphs for aerothermodynamic gas turbine stage topology,
sensor measurement routing, and fleet-level operational wear similarity networks.
"""

import logging
from typing import Dict, Any, List, Tuple
import networkx as nx
import numpy as np
import pandas as pd

logger = logging.getLogger("AURA-M.GraphIntelligence")

class TurbofanGraphEngine:
    """Constructs physical gas turbine stage topology and fleet-level similarity networks."""

    @classmethod
    def build_aerothermodynamic_stage_graph(cls) -> nx.DiGraph:
        """
        Constructs directed graph representing the physical flow of gas through the turbofan sections:
        Airflow: Air Inlet -> Fan -> [LPC -> HPC -> Combustor -> HPT -> LPT] & [Bypass Duct -> Mixer] -> Exhaust
        Mechanical Shafts:
        - Low-Pressure Shaft connects Fan, LPC, and LPT
        - High-Pressure Shaft connects HPC and HPT
        """
        G = nx.DiGraph(name="Turbofan_Gas_Path_Topology")

        # Nodes: Stages
        stages = [
            ("Inlet", {"type": "inlet", "pressure_station": "P2", "temperature_station": "T2"}),
            ("Fan", {"type": "turbomachinery", "spool": "LP", "sensors": ["s_1", "s_8", "s_13"]}),
            ("Bypass_Duct", {"type": "duct", "pressure_station": "P15", "sensors": ["s_6", "s_15"]}),
            ("LPC", {"type": "compressor", "spool": "LP", "pressure_station": "P24", "sensors": ["s_2"]}),
            ("HPC", {"type": "compressor", "spool": "HP", "pressure_station": "P30", "sensors": ["s_3", "s_7", "s_11"]}),
            ("Combustor", {"type": "burner", "sensors": ["s_4", "s_12", "s_16"]}),
            ("HPT", {"type": "turbine", "spool": "HP", "sensors": ["s_9", "s_14", "s_20"]}),
            ("LPT", {"type": "turbine", "spool": "LP", "sensors": ["s_17", "s_21"]}),
            ("Exhaust_Nozzle", {"type": "exhaust"})
        ]
        for node, attrs in stages:
            G.add_node(node, **attrs)

        # Aerodynamic Fluid Flow Edges
        gas_edges = [
            ("Inlet", "Fan", {"type": "fluid_flow"}),
            ("Fan", "Bypass_Duct", {"type": "fluid_flow", "ratio": "bypass"}),
            ("Fan", "LPC", {"type": "fluid_flow", "ratio": "core"}),
            ("LPC", "HPC", {"type": "fluid_flow"}),
            ("HPC", "Combustor", {"type": "fluid_flow"}),
            ("Combustor", "HPT", {"type": "fluid_flow"}),
            ("HPT", "LPT", {"type": "fluid_flow"}),
            ("LPT", "Exhaust_Nozzle", {"type": "fluid_flow"}),
            ("Bypass_Duct", "Exhaust_Nozzle", {"type": "fluid_flow"})
        ]
        G.add_edges_from([(u, v, d) for u, v, d in gas_edges])

        return G

    @classmethod
    def build_fleet_similarity_graph(
        cls,
        fleet_telemetry: List[Dict[str, Any]],
        similarity_threshold: float = 0.85
    ) -> nx.Graph:
        """
        Builds undirected fleet similarity graph where edges connect engines exhibiting
        similar multi-channel degradation wear vectors.
        """
        G = nx.Graph(name="Fleet_Degradation_Similarity")

        for eng in fleet_telemetry:
            G.add_node(
                eng["engine_id"],
                health_index=eng["health_index"],
                predicted_rul=eng["predicted_rul"],
                health_status=eng["health_status"]
            )

        n = len(fleet_telemetry)
        for i in range(n):
            for j in range(i + 1, n):
                e1 = fleet_telemetry[i]
                e2 = fleet_telemetry[j]
                
                # Similarity based on RUL proximity and Health Index closeness
                rul_diff = abs(e1["predicted_rul"] - e2["predicted_rul"])
                hi_diff = abs(e1["health_index"] - e2["health_index"])
                
                sim = 1.0 - (0.6 * min(1.0, rul_diff / 80.0) + 0.4 * min(1.0, hi_diff / 0.5))
                if sim >= similarity_threshold:
                    G.add_edge(e1["engine_id"], e2["engine_id"], weight=round(sim, 3))

        return G

    @classmethod
    def get_graph_summary(cls, G: nx.Graph) -> Dict[str, Any]:
        """Calculates graph metrics (node count, edge count, density, connected components)."""
        return {
            "name": G.graph.get("name", "Graph"),
            "num_nodes": G.number_of_nodes(),
            "num_edges": G.number_of_edges(),
            "density": round(nx.density(G), 3),
            "is_directed": G.is_directed()
        }
