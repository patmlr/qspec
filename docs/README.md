# Docs

The website is a Jekyll site with the [qdocs](https://github.com/patmlr/qdocs) theme, which it shares with qunits.
Most pages are written by hand. The API pages and the tutorial code blocks are generated from the qspec sources by
[`_gen/build.py`](_gen/build.py), which configures the qdocs generator for qspec.

## Build

```bash
pip install -r docs/requirements.txt  # In an environment where qspec is importable.
python docs/_gen/build.py             # Or one of: api, tutorials, logo.
cd docs
bundle exec jekyll serve -I           # Downloads the theme from GitHub.
```

The generated files are committed, because the GitHub Pages workflow only runs Jekyll.
Run the generator after changing docstrings, signatures or the tutorial examples.

## What is generated

| Target      | Source                                    | Output                                           |
|-------------|-------------------------------------------|--------------------------------------------------|
| `api`       | docstrings and signatures in `qspec/`     | `doc/functions/**`, `_includes/api-nav.html`     |
| `tutorials` | `# docs: <name>` regions in `examples/`   | `_includes/code/<name>.html`                     |
| `logo`      | `assets/img/logo.svg`                     | `assets/img/logo_dark.svg`                       |

Do not edit the outputs by hand, they are overwritten. The module pages `doc/modules/*.html` and `doc/doc.html`
are written by hand and use the `api` layout, which adds the navigation.

The docstring markup and the tutorial code regions are described in the
[qdocs README](https://github.com/patmlr/qdocs#docstrings).

## What is qspec specific

- `_config.yml`: the title, description, `accent_color` and the pinned theme version `remote_theme: patmlr/qdocs@vX.Y.Z`.
- `_data/`: the top navigation and the list of tutorials.
- `_includes/badges.html`: the badges below the description.
- `assets/img/`: the logos and the figures of the pages.

Any other theme file can be overridden by a file at the same path here. The layout, colors and code highlighting are
changed in qdocs, not here.
