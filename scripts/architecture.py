"""Render the openai-cost-guard component architecture diagram.

Models the real components found in the code: an application records usage through
the @track_cost decorator family or the FastAPI middleware; both funnel into the
CostTracker, which prices each call against the pricing table and stores records;
reporters turn that usage into a console table, JSON, or Azure Monitor metrics.

Requires the `diagrams` Python package and the Graphviz `dot` binary.

Run from the project root:

    pip install diagrams
    # Graphviz system binary (provides `dot`):
    #   Debian/Ubuntu: sudo apt-get install graphviz
    #   macOS:         brew install graphviz
    python scripts/architecture.py

Output: assets/architecture.png
"""
from pathlib import Path

from diagrams import Cluster, Diagram, Edge
from diagrams.programming.flowchart import Action, Database, Decision
from diagrams.programming.language import Python

OUTPUT_DIR = Path(__file__).resolve().parent.parent / "assets"
OUTPUT_BASENAME = OUTPUT_DIR / "architecture"

GRAPH_ATTR = {
    "fontsize": "20",
    "labelloc": "t",
    "pad": "0.5",
    "splines": "spline",
}


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    with Diagram(
        "openai-cost-guard",
        filename=str(OUTPUT_BASENAME),
        show=False,
        direction="LR",
        graph_attr=GRAPH_ATTR,
        outformat="png",
    ):
        app = Python("Your application\n(Azure OpenAI calls)")

        with Cluster("Entry points"):
            decorators = Action("@track_cost\ndecorators\n(sync / async / stream)")
            middleware = Action("CostGuardMiddleware\n(FastAPI / Starlette,\nper-request scope)")

        pricing = Database("Pricing table\n(defaults + overrides)")

        with Cluster("Core"):
            tracker = Decision("CostTracker\n(price, record,\nbudget check)")

        with Cluster("Reporters"):
            console = Action("Console table")
            json_out = Database("JSON\n(string / file)")
            azure = Action("Azure Monitor\n(OpenTelemetry)")

        cli = Python("openai-cost-guard CLI\n(show / summary)")

        app >> Edge(label="wraps calls") >> decorators
        app >> Edge(label="HTTP requests") >> middleware

        decorators >> Edge(label="record()") >> tracker
        middleware >> Edge(label="record()") >> tracker

        pricing >> Edge(label="lookup", style="dashed") >> tracker

        tracker >> Edge(label="report()") >> console
        tracker >> Edge(label="report()") >> json_out
        tracker >> Edge(label="on_record hook") >> azure

        json_out >> Edge(label="reads", style="dashed") >> cli


if __name__ == "__main__":
    main()
