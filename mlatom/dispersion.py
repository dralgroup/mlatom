#!/usr/bin/env python3
'''
.. code-block::

  !---------------------------------------------------------------------------!
  ! dispersion: naming, resolving and building dispersion corrections          !
  ! Implementations by: Pavlo O. Dral                                          !
  !---------------------------------------------------------------------------!

The single place where a dispersion term is named, validated and turned into a
model tree node.

The rule the rest of the package relies on is that a dispersion term is
**symmetric**: whatever ``dispersion_kwargs`` names is subtracted from the
reference labels during fine-tuning *and* added back at prediction.  Because it
is the same term on both sides, the reference energies are reproduced whichever
term is chosen - a different choice changes only how much of the model is
analytic and how it extrapolates.

Accepted values of ``dispersion_kwargs``:

* a complete spec, e.g. ``{'method': 'd3bj', 'functional': 'b3lyp'}`` - that term
  is subtracted from the labels and is the dispersion node of the final model;
* ``False`` - no dispersion at all: nothing subtracted, no node;
* ``{}`` - raises, because it means "off" for AIQM1/2/3 but "keep the library
  entry" for UAIQM and so has no single meaning.

``method`` is required.  ``{'functional': 'b3lyp'}`` alone selects D3 or D4
depending only on which model class it is passed to, and within D3 does not say
which damping form is meant, so an underdetermined spec could subtract
``D3(BJ)/b3lyp`` and add ``D4/b3lyp`` while reporting that both are "b3lyp".
'''

import os
import subprocess

from . import universal_mlips

# Dispersion programs reachable from MLatom, and the damping forms each offers.
d3_methods = ['d3zero', 'd3bj', 'd3bjm', 'd3zerom', 'd3op']
d4_methods = ['d4']
supported_methods = d3_methods + d4_methods

# Keys that identify the term itself, as opposed to how it is run.  Only these
# go into a provenance stamp; working_directory and friends do not.
identity_keys = ('method', 'functional', 'damping_function_params')

# The dispersion term each pretrained universal model was built with.  A model
# with a native term requires an explicit choice at train() time; one without has
# nothing to choose.  Keys are casefolded method names.


class dispersion_error(ValueError):
    '''Raised when a dispersion declaration is missing, incomplete or contradictory.'''
    pass


def native_dispersion_kwargs(method):
    '''
    The dispersion term ``method`` was originally built with, or ``None`` if it
    has none (``@DFT*`` variants, plain ANI, OMNI-P2x).

    ANI is decided by its method name: only the ``-D4`` variants carry a term.
    '''
    if method is None:
        return None
    name = str(method).casefold()
    if universal_mlips.entry(name) is not None:
        return universal_mlips.native_dispersion(name)
    if name.startswith('ani'):
        if 'd4' in name:
            # Same pinned damping as the AIQM/OMNI-P entries - see
            # universal_mlips.D4_WB97X for why the parameters are named rather
            # than the functional.
            return {'method': 'd4', 'functional': 'wb97x',
                    'damping_function_params': list(universal_mlips.D4_WB97X)}
        return None
    return None


def has_native_dispersion(method):
    '''Whether ``method`` has a dispersion term of its own, i.e. something to choose.'''
    return native_dispersion_kwargs(method) is not None


def describe(dispersion_kwargs):
    '''
    Human-readable name of a dispersion term, e.g. ``D4(wb97x)`` or
    ``D3BJ(b3lyp)``.  Used in printed energy expressions and error messages.
    '''
    if not dispersion_kwargs:
        return 'none'
    spec = dict(dispersion_kwargs)
    method = str(spec.get('method', '')).casefold()
    functional = spec.get('functional', None)
    label = 'D4' if method in d4_methods else method.upper()
    if spec.get('damping_function_params', None):
        params = ','.join(str(p) for p in spec['damping_function_params'])
        return f'{label}({functional}; params {params})' if functional else f'{label}(params {params})'
    return f'{label}({functional})' if functional else label


def normalize(dispersion_kwargs, context=''):
    '''
    Validate a dispersion spec and return it in canonical form, or ``None`` when
    it says "no dispersion".

    ``False``/``None`` -> ``None``; ``{}`` raises; a dict must carry ``method``.
    '''
    if dispersion_kwargs is None or dispersion_kwargs is False:
        return None
    if not isinstance(dispersion_kwargs, dict):
        raise dispersion_error(
            f"dispersion_kwargs must be a dict or False, got {type(dispersion_kwargs).__name__}. "
            f"Example: {{'method': 'd3bj', 'functional': 'b3lyp'}}, or False for no dispersion."
        )
    if len(dispersion_kwargs) == 0:
        raise dispersion_error(
            "dispersion_kwargs={} has no single meaning - it reads as 'off' for AIQM1/2/3 "
            "but as 'keep the library entry' for UAIQM. Write False to switch dispersion off, "
            "or name the term, e.g. {'method': 'd4', 'functional': 'wb97x'}."
            + (f' ({context})' if context else '')
        )
    spec = dict(dispersion_kwargs)
    if 'method' not in spec or not spec['method']:
        raise dispersion_error(
            "dispersion_kwargs needs a 'method': 'functional' alone does not say whether D3 or D4 "
            "is meant, nor which D3 damping form, so the term subtracted from the labels could "
            "differ from the one added back. "
            f"Supported methods: {supported_methods}. "
            f"Example: {{'method': 'd3bj', 'functional': {spec.get('functional', 'b3lyp')!r}}}."
            + (f' ({context})' if context else '')
        )
    spec['method'] = str(spec['method']).casefold()
    if spec['method'] not in supported_methods:
        raise dispersion_error(
            f"unknown dispersion method {spec['method']!r}; supported: {supported_methods}. "
            "MLatom reaches only dftd3 and dftd4; D2, VV10/-V, MBD and TS are not available."
            + (f' ({context})' if context else '')
        )
    return spec


def identity(dispersion_kwargs):
    '''
    The part of a spec that identifies the term, dropping run-time settings such
    as ``working_directory``.  This is what a provenance stamp records and what
    two declarations are compared on.
    '''
    spec = normalize(dispersion_kwargs)
    if spec is None:
        return None
    return {key: spec[key] for key in identity_keys if key in spec and spec[key] is not None}


def same(one, other):
    '''Whether two dispersion declarations name the same term.'''
    return identity(one) == identity(other)


def missing_declaration_error(method):
    '''
    The error raised when ``train()`` is called without a dispersion declaration
    on a model that has a native term.  There is deliberately no default: any
    default hides a term from someone - defaulting to the native term silently
    adds D4 for a user fine-tuning on plain wB97X, defaulting to none silently
    removes the long-range tail for a user fine-tuning on MP2.
    '''
    native = native_dispersion_kwargs(method)
    return dispersion_error(
        f"{method}.train() requires dispersion_kwargs=, or a delta database carrying a "
        f"provenance stamp.\n"
        f"  Whatever you name here is subtracted from your reference labels AND added back\n"
        f"  at prediction, so your energies are reproduced either way.\n"
        f"    {native!r}\n"
        f"        native to {method} - use when your level's dispersion cannot be\n"
        f"        cleanly separated (MP2, CCSD(T))\n"
        f"    {{'method': 'd3bj', 'functional': 'b3lyp'}}\n"
        f"        labels are B3LYP-D3(BJ) or similar\n"
        f"    False\n"
        f"        no dispersion in the labels and none in the model (plain B3LYP, plain wB97X)\n"
        f"  'method' is not optional: 'functional' alone does not say D3 or D4, nor which\n"
        f"  D3 damping form."
    )


def resolve(method, dispersion_kwargs='not given', stamp=None, required=None):
    '''
    Work out which dispersion term applies, from the two places a declaration may
    come from - never neither, and never two that disagree.

    Arguments:
        method (str): the model's method name, used for the native term and messages.
        dispersion_kwargs: the value passed by the user; the sentinel string
            ``'not given'`` means the keyword was omitted.
        stamp (dict, optional): provenance of a pre-computed delta database.
        required (bool, optional): whether a declaration is mandatory; defaults to
            whether ``method`` has a native dispersion term.

    Returns:
        The canonical spec, or ``None`` for no dispersion.
    '''
    if required is None:
        required = has_native_dispersion(method)

    given = dispersion_kwargs != 'not given'
    from_stamp = None
    if stamp:
        from_stamp = stamp.get('dispersion', None)

    if given:
        spec = normalize(dispersion_kwargs, context=f'in {method}.train()')
        if stamp is not None and not same(spec, from_stamp):
            raise dispersion_error(
                f"dispersion_kwargs={dispersion_kwargs!r} does not match the delta database, "
                f"whose labels already had {describe(from_stamp)} subtracted. "
                f"They must name the same term, otherwise the model would add back something "
                f"other than what was removed. Drop dispersion_kwargs to use the database's own "
                f"declaration, or prepare the labels again with the term you want."
            )
        return spec

    if stamp is not None:
        # The decision was made and recorded when the labels were built.
        return normalize(from_stamp)

    if required:
        raise missing_declaration_error(method)
    return None


def program_version(dispersion_kwargs):
    '''
    Version string of the dispersion program that would evaluate this term, so a
    stamp records not just the parameters but the parameterisation.  Returns
    ``None`` if the program cannot be queried.
    '''
    spec = normalize(dispersion_kwargs)
    if spec is None:
        return None
    binary = os.environ.get('dftd4bin' if spec['method'] in d4_methods else 'dftd3bin', None)
    if not binary:
        return None
    try:
        completed = subprocess.run([binary, '--version'], capture_output=True,
                                   universal_newlines=True, timeout=60)
        output = (completed.stdout or completed.stderr).strip().splitlines()
        return output[0].strip() if output else None
    except Exception:
        return None


def build_node(dispersion_kwargs, name='dispersion', working_directory=None):
    '''
    Build the model tree node for a dispersion term, or return ``None`` when
    there is no dispersion.

    ``name`` is the attribute the term lands on, ``mol.<name>.energy``. Each
    family passes the spelling it has always used - ``d4wb97x`` for AIQM2 and
    OMNI-P1, ``d4_wb97x`` for AIQM1, ``d3b973c`` for AIQM3 - because those are
    published properties that user scripts and the shipped tests read. A uniform
    ``dispersion`` would be tidier internally and was tried, but renaming them
    broke ``mol.d4wb97x.energy``; the uniform name is used only for the label
    source in a prepared training database, which is new surface.
    '''
    spec = normalize(dispersion_kwargs)
    if spec is None:
        return None
    from .models import model_tree_node, methods
    kwargs = dict(spec)
    method = kwargs.pop('method')
    if working_directory is not None:
        kwargs['working_directory'] = working_directory
    return model_tree_node(
        name=name,
        model=methods(method=method, **kwargs),
        operator='predict',
    )


def make_stamp(baseline=None, dispersion_kwargs=None, gradients=False,
               target_property='energy', method=None):
    '''
    Provenance of a prepared database: what was subtracted from the reference
    labels, with which program, and whether gradients were subtracted too.

    ``train()`` reads this instead of being told a second time, which is what
    makes user-side label preparation safe: the hazard was never the user
    preparing labels, it was ``train()`` receiving labels of unknown provenance.
    '''
    spec = identity(dispersion_kwargs)
    return {
        'method': method,
        'baseline': baseline,
        'dispersion': spec,
        'dispersion_program_version': program_version(dispersion_kwargs) if spec else None,
        'gradients_subtracted': bool(gradients),
        'target_property': target_property,
    }


def energy_expression(baseline=None, neural_network='NN', dispersion_kwargs=None):
    '''
    The composite energy expression, e.g. ``E = GFN2-xTB* + dNN + D4(wb97x)``.
    Printed at the end of train() and stored in tree.json, because every defect
    in this area is invisible at the call site.
    '''
    terms = []
    if baseline:
        terms.append(str(baseline))
    terms.append(neural_network)
    if identity(dispersion_kwargs):
        terms.append(describe(dispersion_kwargs))
    return 'E = ' + ' + '.join(terms)
