"""Smoke tests that do not call external APIs."""

import unittest


class TestCompetitorIntelGraph(unittest.TestCase):
    def test_graph_compiles(self) -> None:
        from competitor_intel.graph.workflow import build_competitor_intel_graph

        graph = build_competitor_intel_graph()
        self.assertIsNotNone(graph)


if __name__ == "__main__":
    unittest.main()
