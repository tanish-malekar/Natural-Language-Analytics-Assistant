from decimal import Decimal
from numbers import Number
from pathlib import Path
import pandas as pd
import plotly.express as px


def render_chart(result, specification, output):
    if specification["kind"] == "none" or not result.rows:
        return None
    frame = pd.DataFrame(result.rows, columns=result.columns)
    if frame.columns.duplicated().any():
        raise ValueError("Charting requires unique selected aliases")
    lookup = {str(column).lower(): column for column in frame.columns}
    if specification["kind"] == "auto":
        if len(frame.columns) < 2:
            return None
        numeric = [column for column in frame.columns if frame[column].dropna().map(lambda value: isinstance(value, (Number, Decimal)) and not isinstance(value, bool)).all()]
        if not numeric:
            return None
        y = numeric[-1]
        x = next((column for column in frame.columns if column != y), None)
        if x is None:
            return None
        temporal = any(word in str(x).upper() for word in ("DATE", "MONTH", "YEAR"))
        specification = {**specification, "kind": "line" if temporal else "bar", "x_column": str(x), "y_column": str(y)}
    requested = [specification.get("x_column"), specification.get("y_column")]
    if any(name is None or name.lower() not in lookup for name in requested):
        raise ValueError("Chart aliases must be present in the query result")
    x, y = [lookup[name.lower()] for name in requested]
    if not frame[y].dropna().map(lambda value: isinstance(value, (Number, Decimal)) and not isinstance(value, bool)).all():
        raise ValueError("Chart y-axis must contain numeric values")
    frame[y] = pd.to_numeric(frame[y])
    functions = {"bar": px.bar, "line": px.line, "scatter": px.scatter}
    if specification["kind"] not in functions:
        raise ValueError("Unsupported chart kind")
    figure = functions[specification["kind"]](frame, x=x, y=y, title=specification["title"], template="plotly_white")
    figure.update_layout(margin=dict(l=40, r=30, t=65, b=40))
    path = Path(output) / "chart.html"
    figure.write_html(path, include_plotlyjs=True, full_html=True)
    return str(path)
