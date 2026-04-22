"""Step 05: LP counterfactual via the patched extract_weff API.

PATCH-BLOCKED: requires the extract_weff kwarg on
KnowledgeMatrixComputer.forward(). See ../knowledgematrix/patch.md.

Run this script to verify the patch is available before submitting cluster jobs.
"""
import inspect
import sys


def patch_available() -> bool:
    """Return True iff KnowledgeMatrixComputer.forward accepts extract_weff."""
    from knowledgematrix.matrix_computer import KnowledgeMatrixComputer

    sig = inspect.signature(KnowledgeMatrixComputer.forward)
    return "extract_weff" in sig.parameters


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
    print("Patch available — implementation pending (Task 17).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
