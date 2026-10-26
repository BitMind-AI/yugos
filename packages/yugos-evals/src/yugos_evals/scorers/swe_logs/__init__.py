from .python import (
    parse_log_pytest,
    parse_log_pytest_options,
    parse_log_django,
    parse_log_pytest_v2,
    parse_log_seaborn,
    parse_log_sympy,
    parse_log_matplotlib,
    parse_log_astroid,
    parse_log_flask,
    parse_log_marshmallow,
    parse_log_pvlib,
    parse_log_pyvista,
    parse_log_sqlfluff,
    parse_log_xarray,
    parse_log_pydicom,
    parse_log_requests,
    parse_log_pylint,
    parse_log_astropy,
    parse_log_scikit,
    parse_log_sphinx,
)
from .javascript import (
    parse_log_calypso,
    parse_log_chart_js,
    parse_log_marked,
    parse_log_p5js,
    parse_log_react_pdf,
    parse_log_jest,
    parse_log_jest_json,
    parse_log_vitest,
    parse_log_karma,
    parse_log_tap,
    parse_log_immutable_js,
    parse_log_bpmn_js,
    parse_log_carbon,
    parse_log_eslint,
    parse_log_grommet,
    parse_log_highlightjs,
    parse_log_lighthouse,
    parse_log_next,
    parse_log_openlayers,
    parse_log_prismjs,
    parse_log_quarto_cli,
)
from .c import (
    parse_log_redis,
    parse_log_jq,
    parse_log_doctest,
    parse_log_micropython_test,
    parse_log_googletest,
)
from .go import parse_log_gotest
from .java import (
    parse_log_maven,
    parse_log_ant,
    parse_log_gradle_custom,
)
from .php import parse_log_phpunit
from .ruby import (
    parse_log_minitest,
    parse_log_cucumber,
    parse_log_ruby_unit,
    parse_log_rspec_transformed_json,
    parse_log_jekyll,
)
from .rust import parse_log_cargo

PARSER_REGISTRY = {
    # Python
    "parse_log_pytest": parse_log_pytest,
    "parse_log_pytest_options": parse_log_pytest_options,
    "parse_log_django": parse_log_django,
    "parse_log_pytest_v2": parse_log_pytest_v2,
    "parse_log_seaborn": parse_log_seaborn,
    "parse_log_sympy": parse_log_sympy,
    "parse_log_matplotlib": parse_log_matplotlib,
    "parse_log_astroid": parse_log_astroid,
    "parse_log_flask": parse_log_flask,
    "parse_log_marshmallow": parse_log_marshmallow,
    "parse_log_pvlib": parse_log_pvlib,
    "parse_log_pyvista": parse_log_pyvista,
    "parse_log_sqlfluff": parse_log_sqlfluff,
    "parse_log_xarray": parse_log_xarray,
    "parse_log_pydicom": parse_log_pydicom,
    "parse_log_requests": parse_log_requests,
    "parse_log_pylint": parse_log_pylint,
    "parse_log_astropy": parse_log_astropy,
    "parse_log_scikit": parse_log_scikit,
    "parse_log_sphinx": parse_log_sphinx,
    # JavaScript
    "parse_log_calypso": parse_log_calypso,
    "parse_log_bpmn_js": parse_log_bpmn_js,
    "parse_log_carbon": parse_log_carbon,
    "parse_log_eslint": parse_log_eslint,
    "parse_log_grommet": parse_log_grommet,
    "parse_log_highlightjs": parse_log_highlightjs,
    "parse_log_lighthouse": parse_log_lighthouse,
    "parse_log_next": parse_log_next,
    "parse_log_openlayers": parse_log_openlayers,
    "parse_log_prismjs": parse_log_prismjs,
    "parse_log_quarto_cli": parse_log_quarto_cli,
    "parse_log_chart_js": parse_log_chart_js,
    "parse_log_marked": parse_log_marked,
    "parse_log_p5js": parse_log_p5js,
    "parse_log_react_pdf": parse_log_react_pdf,
    "parse_log_jest": parse_log_jest,
    "parse_log_jest_json": parse_log_jest_json,
    "parse_log_vitest": parse_log_vitest,
    "parse_log_karma": parse_log_karma,
    "parse_log_tap": parse_log_tap,
    "parse_log_immutable_js": parse_log_immutable_js,
    # C
    "parse_log_redis": parse_log_redis,
    "parse_log_jq": parse_log_jq,
    "parse_log_doctest": parse_log_doctest,
    "parse_log_micropython_test": parse_log_micropython_test,
    "parse_log_googletest": parse_log_googletest,
    # Go
    "parse_log_gotest": parse_log_gotest,
    # Java
    "parse_log_maven": parse_log_maven,
    "parse_log_ant": parse_log_ant,
    "parse_log_gradle_custom": parse_log_gradle_custom,
    # PHP
    "parse_log_phpunit": parse_log_phpunit,
    # Ruby
    "parse_log_minitest": parse_log_minitest,
    "parse_log_cucumber": parse_log_cucumber,
    "parse_log_ruby_unit": parse_log_ruby_unit,
    "parse_log_rspec_transformed_json": parse_log_rspec_transformed_json,
    "parse_log_jekyll": parse_log_jekyll,
    # Rust
    "parse_log_cargo": parse_log_cargo,
}


__all__ = [
    "PARSER_REGISTRY",
]
