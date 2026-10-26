"""
Generates the generated parts of the qspec website with qdocs (https://github.com/patmlr/qdocs). Run from anywhere:

    python docs/_gen/build.py [api | tutorials | logo | all]

- api: the API pages in docs/doc/functions/ and the navigation docs/_includes/api-nav.html.
- tutorials: the highlighted code blocks docs/_includes/code/*.html from the marked regions in examples/*.py.
- logo: the dark mode logo docs/assets/img/logo_dark.svg from logo.svg.
"""

import os

from qdocs import Project, main

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

PROJECT = Project(
    package="qspec",
    modules=["qtypes", "physics", "algebra", "models", "analyze", "simulate", "tools", "stats"],
    root=ROOT,
    namespaced={"models", "simulate", "analyze"},
    include={
        "qtypes": {
            "HasGetItem",
            "HasShape",
            "asarray",
            "asarray_none",
            "cast",
            "cast_sympy",
            "has_getitem",
            "has_shape",
            "is_scalar",
            "is_sympy_expr",
            "quant",
        }
    },
)

if __name__ == "__main__":
    main(PROJECT)
