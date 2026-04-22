"""Step 06: Jacobian sensitivity heatmap from W_eff.

PATCH-BLOCKED: requires the extract_weff kwarg.
See ../knowledgematrix/patch.md.
"""
import sys

from km_feature_viz.counterfactual_lp import patch_available


def main() -> int:
    if not patch_available():
        print(
            "ERROR: knowledgematrix is not patched yet.\n"
            "  Required: KnowledgeMatrixComputer.forward(x, extract_weff=True)\n"
            "  See ../knowledgematrix/patch.md for the brief.\n"
            "  This script will be implemented after the patch lands.",
            file=sys.stderr,
        )
        return 2
    print("Patch available — implementation pending (Task 18).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
