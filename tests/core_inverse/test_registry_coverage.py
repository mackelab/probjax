from probjax.core.registry import Context

from . import (
    test_array_rules,
    test_control_flow,
    test_dot_general,
    test_partial_propagation,
)
from .cases import (
    ATOMIC_ROUNDTRIP_CASES,
    LOGDET_EXPLICIT_CASES,
    MANUAL_COVERAGE,
    PRINCIPAL_BRANCH_CASES,
    STRUCTURALLY_INVALID_CASES,
    registered_inverse_primitives,
    registered_logdet_primitives,
)


_MANUAL_MODULES = {
    "array": test_array_rules,
    "control": test_control_flow,
    "dot": test_dot_general,
    "partial": test_partial_propagation,
}


def test_manual_coverage_points_to_collected_tests():
    for case in MANUAL_COVERAGE:
        test = getattr(_MANUAL_MODULES[case.module], case.test_name, None)
        assert callable(test), f"Missing manual coverage test {case.module}.{case.test_name}"
        assert case.test_name.startswith("test_")


def test_all_inverse_primitives_have_executable_cases():
    declarative = {
        case.primitive
        for case in (
            ATOMIC_ROUNDTRIP_CASES
            + PRINCIPAL_BRANCH_CASES
            + STRUCTURALLY_INVALID_CASES
        )
    }
    manual = {
        case.primitive for case in MANUAL_COVERAGE if case.context == Context.INVERSE
    }
    assert registered_inverse_primitives() == declarative | manual


def test_all_inverse_logdet_primitives_have_executable_cases():
    declarative = {case.primitive for case in LOGDET_EXPLICIT_CASES}
    manual = {
        case.primitive
        for case in MANUAL_COVERAGE
        if case.context == Context.INVERSE_LOGDET
    }
    assert registered_logdet_primitives() == declarative | manual
