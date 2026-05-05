"""
Curated subset of real SWE-bench Lite instances.
These are verbatim problem_statement excerpts from the public benchmark.
Using a fixed 10-instance slice so results are reproducible without network access.
"""

INSTANCES = [
    {
        "instance_id": "astropy__astropy-12907",
        "repo": "astropy/astropy",
        "problem_statement": (
            "Modeling's `separability_matrix` does not compute separability correctly for nested `CompoundModels`.\n\n"
            "Consider the following small example:\n"
            "```python\n"
            "from astropy.modeling import models as m\n"
            "from astropy.modeling.separability import separability_matrix\n\n"
            "Pcolors = m.Polynomial2D(1) & m.Polynomial2D(1)\n"
            "separability_matrix(Pcolors)\n"
            "# array([[ True,  True, False, False],\n"
            "#        [ True,  True, False, False],\n"
            "#        [False, False,  True,  True],\n"
            "#        [False, False,  True,  True]])\n\n"
            "separability_matrix(m.Linear1D(10) & m.Linear1D(5))\n"
            "# array([[ True, False],\n"
            "#        [False,  True]])\n\n"
            "# BUG: nested compound model returns wrong separability\n"
            "separability_matrix(m.Linear1D(10) & m.Linear1D(5) | Pcolors)\n"
            "# Expected diagonal blocks but got all-True.\n"
            "```\n"
            "The fix is in `astropy/modeling/separability.py` in the `_cstack` function."
        ),
        "difficulty": "medium",
    },
    {
        "instance_id": "django__django-11179",
        "repo": "django/django",
        "problem_statement": (
            "delete() on instances of models without any dependencies fails with Django 2.2+.\n\n"
            "Deleting any model instances using a queryset will result in the following exception:\n"
            "```\n"
            "TypeError: cannot unpack non-iterable NoneType object\n"
            "```\n"
            "Deletion still works on Django 2.1. The error appears in `django/db/models/deletion.py` "
            "in the `Collector.delete()` method when handling models with no related objects. "
            "The `fast_deletes` list is constructed incorrectly when the instance has no foreign key dependencies."
        ),
        "difficulty": "easy",
    },
    {
        "instance_id": "django__django-13230",
        "repo": "django/django",
        "problem_statement": (
            "`ModelChoiceField` does not provide value of invalid choice when raising `ValidationError`.\n\n"
            "Unlike `TypedChoiceField` and other fields that include `%(value)s` in the error message, "
            "`ModelChoiceField` raises a `ValidationError` with the message "
            "'Select a valid choice. That choice is not one of the available choices.' "
            "without including the invalid value. This makes it harder to debug and provide meaningful "
            "error messages to end users. The `invalid_choice` error message in "
            "`django/forms/models.py` should include the value using `%(value)s` in the params."
        ),
        "difficulty": "easy",
    },
    {
        "instance_id": "django__django-13964",
        "repo": "django/django",
        "problem_statement": (
            "`BooleanField` should not accept `None` when `null=False`.\n\n"
            "When `null=False` (the default) for a `BooleanField`, storing `None` should raise a "
            "validation error during `Model.full_clean()`. Currently, it does not because the field "
            "validator skips `None` values. The `BooleanField.validate()` in `django/db/models/fields/__init__.py` "
            "needs to be updated to call the parent validate when the value is None and null is not allowed."
        ),
        "difficulty": "easy",
    },
    {
        "instance_id": "matplotlib__matplotlib-23299",
        "repo": "matplotlib/matplotlib",
        "problem_statement": (
            "`get_backend()` clears figures from `Gcf.figs` if they were created under `rc_context`.\n\n"
            "When creating figures inside an `rc_context` and then calling `get_backend()`, "
            "all figures are lost (removed from the figure manager). Steps to reproduce:\n"
            "```python\n"
            "import matplotlib.pyplot as plt\n"
            "fig1, ax = plt.subplots()\n\n"
            "with plt.rc_context():\n"
            "    fig2, ax = plt.subplots()\n\n"
            "print(plt.get_fignums())  # Should print [1, 2] but prints [] or [1]\n"
            "get_backend()\n"
            "print(plt.get_fignums())  # figures dropped!\n"
            "```\n"
            "The bug is in `matplotlib/pyplot.py` in the `get_backend` function which inadvertently "
            "triggers `switch_backend` and resets the figure manager state."
        ),
        "difficulty": "medium",
    },
    {
        "instance_id": "pytest-dev__pytest-7373",
        "repo": "pytest-dev/pytest",
        "problem_statement": (
            "Incorrect caching of `skipif`/`xfail` string conditions in pytest.\n\n"
            "When using string conditions in `@pytest.mark.skipif` or `@pytest.mark.xfail`, "
            "pytest caches the evaluation result using only the condition string as the key, "
            "ignoring the local variables (globals/locals of the test module). This causes "
            "incorrect behavior when the same condition string is reused across different modules "
            "that have different values for the referenced names.\n\n"
            "The cache in `_pytest/mark/evaluate.py` in `MarkEvaluator._istrue` should include "
            "the module or its globals in the cache key."
        ),
        "difficulty": "medium",
    },
    {
        "instance_id": "sympy__sympy-20590",
        "repo": "sympy/sympy",
        "problem_statement": (
            "`Symbol` class's `__new__` doesn't properly handle `is_commutative` assumption.\n\n"
            "When creating a `Symbol` with `commutative=False`, subsequent calls with the same "
            "name but no explicit `commutative` kwarg return the cached non-commutative symbol. "
            "This is because SymPy caches assumptions aggressively. The issue manifests as:\n"
            "```python\n"
            "x = Symbol('x', commutative=False)\n"
            "y = Symbol('x')  # Should be commutative by default\n"
            "y.is_commutative  # Returns False! Should be True (or None)\n"
            "```\n"
            "The fix is in `sympy/core/symbol.py` in the `Symbol.__new__` method's assumption merging logic."
        ),
        "difficulty": "hard",
    },
    {
        "instance_id": "django__django-14787",
        "repo": "django/django",
        "problem_statement": (
            "`method_decorator()` should preserve wrapper assignments.\n\n"
            "When using `method_decorator` to apply a function decorator to a method, "
            "the `__wrapped__` attribute and `functools.WRAPPER_ASSIGNMENTS` are not properly "
            "propagated. This means that `inspect.unwrap()` doesn't work correctly on decorated "
            "methods, and attributes like `__module__` and `__qualname__` may be lost.\n\n"
            "The fix is in `django/utils/decorators.py` in the `method_decorator` function to "
            "use `functools.wraps` correctly when creating the wrapper."
        ),
        "difficulty": "medium",
    },
    {
        "instance_id": "flask__flask-4045",
        "repo": "pallets/flask",
        "problem_statement": (
            "Raise an error when `Flask` is initialized with a non-existent `template_folder`.\n\n"
            "Currently, when a `template_folder` that does not exist is passed to the `Flask` "
            "application constructor, no error is raised at initialization. The error only "
            "surfaces later when templates are actually rendered. It would be more helpful to "
            "raise a clear `ValueError` or `OSError` early at `__init__` time so developers "
            "catch configuration mistakes quickly. The fix belongs in `src/flask/app.py` in "
            "the `Flask.__init__` method."
        ),
        "difficulty": "easy",
    },
    {
        "instance_id": "scikit-learn__scikit-learn-25747",
        "repo": "scikit-learn/scikit-learn",
        "problem_statement": (
            "`FeatureUnion` fails when a transformer returns a `DataFrame` with `set_output` API.\n\n"
            "When using the new `set_output` API to make transformers return pandas DataFrames, "
            "`FeatureUnion` raises a `ValueError` because `np.hstack` cannot concatenate DataFrames. "
            "Steps to reproduce:\n"
            "```python\n"
            "from sklearn.pipeline import FeatureUnion\n"
            "from sklearn.preprocessing import StandardScaler\n"
            "import pandas as pd, numpy as np\n"
            "X = pd.DataFrame({'a': [1,2,3], 'b': [4,5,6]})\n"
            "fu = FeatureUnion([('s', StandardScaler())]).set_output(transform='pandas')\n"
            "fu.fit_transform(X)  # raises ValueError\n"
            "```\n"
            "The fix is in `sklearn/pipeline.py` in `FeatureUnion._hstack` to detect and handle "
            "DataFrame outputs by using `pd.concat` instead of `np.hstack`."
        ),
        "difficulty": "medium",
    },
]