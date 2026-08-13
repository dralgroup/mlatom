#!/usr/bin/env python3
"""
Delta-learning: preparing the data a model is actually fitted to.

Delta learning fits a machine-learning model not to a reference level directly but
to the residual left once the parts of the model that are not trained have been
removed - a baseline method, a dispersion correction, or both. That is a general
idea, older and wider than any particular model family, so this module is written
in those terms: it knows about baselines and dispersion terms, not about which
model happens to use them.

These routines prepare and carry that data for fine-tuning MLatom's universal
models (ANI, AIQM1/2/3, UAIQM, OMNI-P).

The delta labels a fine-tuned model is fitted to are

.. math::
    E_{\\Delta} = E_{\\text{target}} - E_{\\text{baseline}} - E_{\\text{dispersion}}

and the fine-tuned model adds the *same* baseline and the *same* dispersion term
back at prediction.  Because the dispersion term is symmetric, the reference
energies are reproduced whichever term is chosen; the choice changes only how
much of the model is analytic and how it extrapolates.

Every database produced here carries a provenance stamp (see
:mod:`mlatom.dispersion`) recording what was subtracted, with which program, and
whether gradients were subtracted too.  ``train()`` reads that stamp rather than
being told a second time, which is what makes preparing labels by hand safe.
"""
import os
import re

import numpy as np
from . import data
from . import universal_mlips
from . import dispersion as dispersion_utils


def stamp_of(database):
    """
    The provenance stamp of a prepared database, or ``None`` if it has none.

    The stamp and the ``'delta'`` entry of the database's label-source registry
    are the same object - what was subtracted, from what, with which program - so
    either one is enough. Both are written; a database that has only the registry
    is read from there.
    """
    if database is None:
        return None
    stamp = getattr(database, 'provenance', None)
    if stamp:
        return stamp
    registry = getattr(database, 'label_sources', None) or {}
    entry = registry.get('delta', None)
    if entry:
        return {key: value for key, value in entry.items() if key != 'role'}
    return None


def require_stamp(database, keyword='delta_db'):
    """
    Return the provenance stamp of ``database``, refusing one that has none.

    An unstamped database has unknowable provenance, so the only alternatives to
    failing are to guess or to accept an assertion nobody can check - and being
    wrong means the model adds back a dispersion term that was never removed.
    """
    stamp = stamp_of(database)
    if not stamp:
        raise ValueError(
            f"{keyword}= carries no provenance stamp, so MLatom cannot tell what\n"
            f"  was already subtracted from these labels. Re-run prepare_delta_database();\n"
            f"  if you still have the baseline, pass it there as baseline_db= so it is not\n"
            f"  recomputed."
        )
    return stamp


def compute_baseline_database(target_db, baseline_method, baseline_kwargs=None,
                              calculate_energy_gradients=False, verbose=1):
    """Evaluate the frozen baseline over the training geometries."""
    from .models import methods
    if verbose:
        print(f'Computing baseline ({baseline_method}) for delta preparation ...')
    baseline_db = target_db.geometries_only()
    baseline_model = methods(method=baseline_method, **(baseline_kwargs or {}))
    baseline_model.predict(
        molecular_database=baseline_db,
        calculate_energy=True,
        calculate_energy_gradients=calculate_energy_gradients,
    )
    return baseline_db


def compute_dispersion_database(target_db, dispersion_kwargs,
                                calculate_energy_gradients=False,
                                working_directory=None, verbose=1):
    """
    Evaluate the dispersion term over the training geometries.

    This is the term that is subtracted from the reference labels and added back
    by the fine-tuned model - the same object on both sides.
    """
    spec = dispersion_utils.normalize(dispersion_kwargs)
    if spec is None:
        return None
    from .models import methods
    if verbose:
        print(f'Computing dispersion ({dispersion_utils.describe(spec)}) for delta preparation ...')
    dispersion_db = target_db.geometries_only()
    kwargs = dict(spec)
    method = kwargs.pop('method')
    if working_directory is not None:
        kwargs['working_directory'] = working_directory
    dispersion_model = methods(method=method, **kwargs)
    dispersion_model.predict(
        molecular_database=dispersion_db,
        calculate_energy=True,
        calculate_energy_gradients=calculate_energy_gradients,
    )
    return dispersion_db


def require_complete(database, label, with_gradients=False):
    """
    Check that every geometry in a component database actually got a result.

    A baseline or dispersion program that fails on some geometries can leave
    energies missing or NaN. Those would flow straight into the delta labels and
    be trained on, producing a model fitted to nonsense with nothing raised, so
    the failure is reported here instead - naming how many geometries failed and
    where, since on a large set only a handful usually do.
    """
    failed = []
    for imol, mol in enumerate(database.molecules):
        energy = mol.__dict__.get('energy', None)
        try:
            finite = energy is not None and np.isfinite(float(energy))
        except (TypeError, ValueError):
            finite = False
        if not finite:
            failed.append(imol)
            continue
        if with_gradients:
            gradients = np.asarray(
                mol.get_xyz_vectorial_properties('energy_gradients'), dtype=float)
            if gradients.size == 0 or not np.all(np.isfinite(gradients)):
                failed.append(imol)
    if failed:
        shown = ', '.join(str(i) for i in failed[:10])
        more = '' if len(failed) <= 10 else f' (and {len(failed) - 10} more)'
        raise ValueError(
            f"the {label} calculation did not produce a usable result for "
            f"{len(failed)} of {len(database)} geometries: indices {shown}{more}.\n"
            f"  These would otherwise become NaN in the labels and be trained on.\n"
            f"  Check those geometries, then either drop them from the training set or\n"
            f"  compute the {label} yourself and pass it in as "
            f"{'baseline_db=' if label == 'baseline' else 'dispersion_db='}."
        )
    return database


def prepare_delta_database(
    target_db,
    method=None,
    baseline_method=None,
    baseline_kwargs=None,
    baseline_db=None,
    dispersion_kwargs=None,
    dispersion_db=None,
    property_to_learn='energy',
    xyz_derivative_property_to_learn=None,
    delta_property='delta_energy',
    delta_xyz_derivative_property='delta_energy_gradients',
    qm_program=None,
    working_directory=None,
    verbose=1,
    subtract_atomic_shift=False,
):
    """
    Build the delta labels on one database, alongside every component that was
    subtracted to produce them.

    Returns
    -------
    :class:`mlatom.data.molecular_database`
        A copy of ``target_db`` carrying, per molecule: the reference labels in
        the flat slots with ``label_source`` naming where they came from; the
        baseline and the dispersion term under their own names
        (``mol.gfn2xtbstar.energy``, ``mol.dispersion.energy``); and the delta
        labels the network is fitted to.  ``db.label_sources`` says what each of
        those is.
    """
    if not isinstance(target_db, data.molecular_database):
        raise TypeError('target_db must be a molecular_database')

    is_aiqm1 = universal_mlips.has_atomic_shift(method)
    with_gradients = bool(xyz_derivative_property_to_learn)

    # ------------------------------------------------------------------
    # Baseline: given, derived from the method name, or absent (pure NN)
    # ------------------------------------------------------------------
    baseline_db = data.molecular_database.as_database(baseline_db)
    # Derive the baseline name even when the database is supplied, so the
    # provenance stamp records which baseline the labels are relative to.
    if baseline_method is None and method:
        baseline_method = universal_mlips.baseline_method(method)
        baseline_kwargs = dict(baseline_kwargs) if baseline_kwargs else {}
        if qm_program is not None and 'program' not in baseline_kwargs:
            baseline_kwargs['program'] = qm_program
    if baseline_kwargs is None:
        baseline_kwargs = {}
    if working_directory is not None and 'working_directory' not in baseline_kwargs:
        baseline_kwargs = dict(baseline_kwargs, working_directory=working_directory)

    # A previously prepared database carries the baseline under its own name, so
    # baseline_db='my_tl_model/delta_db.h5' reuses what was already computed.
    if baseline_db is not None and baseline_method:
        baseline_db = baseline_db.as_label_source(
            data.molecular_database.source_tag(baseline_method), with_gradients)
    if baseline_db is None and baseline_method is not None:
        baseline_db = compute_baseline_database(
            target_db, baseline_method, baseline_kwargs,
            calculate_energy_gradients=with_gradients, verbose=verbose)
    if baseline_db is not None:
        require_complete(baseline_db, 'baseline', with_gradients)

    # ------------------------------------------------------------------
    # Dispersion: the term named by dispersion_kwargs, subtracted here and
    # added back by the model - the core rule
    # ------------------------------------------------------------------
    dispersion_spec = dispersion_utils.normalize(dispersion_kwargs)
    dispersion_db = data.molecular_database.as_database(dispersion_db)
    if dispersion_db is not None:
        dispersion_db = dispersion_db.as_label_source('dispersion', with_gradients)
    if dispersion_db is None and dispersion_spec is not None:
        dispersion_db = compute_dispersion_database(
            target_db, dispersion_spec,
            calculate_energy_gradients=with_gradients,
            working_directory=working_directory, verbose=verbose)
    if dispersion_db is not None:
        require_complete(dispersion_db, 'dispersion', with_gradients)

    # ------------------------------------------------------------------
    # One database: the reference labels, every component that was subtracted
    # under its own name, and the delta labels the network is fitted to.
    #
    # The components are computed as separate databases because that is how the
    # programs are run, but they are folded onto the geometries here and never
    # kept apart: three databases matched by index is a join with no key, and
    # anything that reorders or subsets one of them pairs the wrong rows.
    # ------------------------------------------------------------------
    delta_db = target_db.copy()
    for mol, target_mol in zip(delta_db.molecules, target_db.molecules):
        mol.id = target_mol.id

    baseline_tag = data.molecular_database.source_tag(baseline_method) if baseline_method else None
    baseline_mols = (baseline_db.aligned_to(delta_db, 'baseline')
                     if baseline_db is not None else None)
    dispersion_mols = (dispersion_db.aligned_to(delta_db, 'dispersion')
                       if dispersion_db is not None else None)

    for imol, mol in enumerate(delta_db.molecules):
        delta_energy = mol.__dict__[property_to_learn]
        delta_grad = (np.array(mol.get_xyz_vectorial_properties(
            xyz_derivative_property_to_learn)) if with_gradients else None)

        for tag, component_mols in ((baseline_tag, baseline_mols),
                                    ('dispersion', dispersion_mols)):
            if component_mols is None:
                continue
            component = component_mols[imol]
            delta_energy = delta_energy - component.__dict__['energy']
            node = data.properties_tree_node(name=tag)
            node.energy = component.__dict__['energy']
            if with_gradients:
                component_grad = component.get_xyz_vectorial_properties('energy_gradients')
                delta_grad = delta_grad - component_grad
                node.energy_gradients = np.asarray(component_grad)
            mol.__dict__[tag] = node

        if is_aiqm1 and subtract_atomic_shift:
            delta_energy -= universal_mlips.atomic_shift_energy(method, mol)
            if with_gradients:
                delta_grad = delta_grad - universal_mlips.atomic_shift_gradients(mol)

        mol.__dict__[delta_property] = delta_energy
        if with_gradients:
            mol.add_xyz_derivative_property(
                delta_grad,
                property_name=delta_property,
                xyz_derivative_property=delta_xyz_derivative_property,
            )

    stamp = dispersion_utils.make_stamp(
        baseline=baseline_method,
        dispersion_kwargs=dispersion_spec,
        gradients=with_gradients,
        target_property=property_to_learn,
        method=method,
    )
    declare_label_sources(delta_db, target_db, stamp,
                           baseline_tag=baseline_tag,
                           baseline_method=baseline_method,
                           dispersion_spec=dispersion_spec,
                           property_to_learn=property_to_learn)
    # Kept alongside the registry entry so databases written before the registry
    # existed still load, and so require_stamp() has one place to look.
    setattr(delta_db, 'provenance', stamp)
    return delta_db


def declare_label_sources(delta_db, target_db, stamp, baseline_tag=None,
                           baseline_method=None, dispersion_spec=None,
                           property_to_learn='energy'):
    """
    Say, in the database itself, what each set of labels on it is.

    The delta's entry is the provenance stamp: what was subtracted, from what,
    with which program.  That is why the registry is of label *sources* and not
    of levels of theory - a delta is not a level, and a registry that could not
    hold one would have had to be joined by a second registry that could.
    """
    target_source = getattr(target_db.molecules[0], 'label_source', None) \
        if len(target_db) else None
    target_source = target_source or 'target'
    inherited = getattr(target_db, 'label_sources', None) or {}
    for name, spec in inherited.items():
        delta_db.label_sources.setdefault(name, dict(spec))
    delta_db.label_sources.setdefault(
        target_source, {'role': 'reference', 'property': property_to_learn})
    for mol in delta_db.molecules:
        mol.__dict__['label_source'] = target_source

    if baseline_tag:
        delta_db.label_sources.setdefault(
            baseline_tag, {'role': 'baseline', 'method': baseline_method})
    if dispersion_spec:
        delta_db.label_sources.setdefault(
            'dispersion', dict(dispersion_spec, role='dispersion'))
    delta_db.label_sources['delta'] = dict(stamp, role='delta')
    return delta_db


def load_delta_database(delta_db):
    """
    Accept a delta database as an object or a filename and return
    ``(database, source_path)``; ``source_path`` is ``None`` for an object.

    The source path matters because a delta database that already exists on disk
    is referenced rather than copied into the model directory.
    """
    if delta_db is None:
        return None, None
    if isinstance(delta_db, str):
        return data.molecular_database.load(delta_db), delta_db
    return data.molecular_database.as_database(delta_db), None


def model_directory(file_to_save_model, default):
    """
    The directory a fine-tuned model's files belong in.

    Most families take ``file_to_save_model`` as a directory; OMNI-P1 takes the
    name of a ``.pt`` file, in which case the directory is the one holding it.
    """
    path = file_to_save_model or default
    if os.path.isdir(path):
        return path
    if os.path.splitext(path)[1]:
        return os.path.dirname(os.path.abspath(path)) or '.'
    return path


def write_training_database(database, directory, delta_source=None, verbose=1):
    """
    Write the prepared training database into the model directory.

    One file, not three.  It carries the reference labels, every component that
    was subtracted under its own name, and the delta labels - so the baseline,
    which is the expensive part of preparation, is kept without a second file and
    without a join to get it back.  Trying a different dispersion choice on the
    same data stays cheap for the same reason.

    It is written whether or not it was asked for: a user who forgets to save it
    pays for the baseline again on the next run, while writing costs nothing.
    Nothing is written twice - a database that came from a file is recorded by
    path instead of being copied.
    """
    written = {}
    if directory is None or database is None:
        return written
    os.makedirs(directory, exist_ok=True)
    if delta_source:
        written['delta'] = os.path.abspath(delta_source)
    else:
        path = os.path.join(directory, 'delta_db.h5')
        database.dump(path, format='h5')
        written['delta'] = path
    if verbose:
        print(f"  training labels: {written['delta']}")
        sources = getattr(database, 'label_sources', None)
        if sources:
            print(f"  label sources:   {', '.join(sorted(sources))}")
    return written


def resolve_training_labels(
    model_method,
    molecular_database=None,
    delta_db=None,
    dispersion_kwargs='not given',
    baseline_method=None,
    baseline_kwargs=None,
    baseline_db=None,
    dispersion_db=None,
    qm_program=None,
    property_to_learn='energy',
    xyz_derivative_property_to_learn=None,
    delta_property='delta_energy',
    delta_xyz_derivative_property='delta_energy_gradients',
    working_directory=None,
    subtract_atomic_shift=False,
    verbose=1,
):
    """
    Work out the labels a fine-tuning run should be fitted to, and the dispersion
    term the resulting model must add back.

    The dispersion declaration comes from exactly one of two places - the
    ``dispersion_kwargs`` keyword, or the provenance stamp of a pre-computed
    ``delta_db`` - never neither, and never two that disagree.

    Returns
    -------
    dict with ``'database'`` (the one prepared database), ``'dispersion_kwargs'``
    (the resolved spec, or ``None``), ``'stamp'`` and ``'delta_source'``.
    """
    delta_database, delta_source = load_delta_database(delta_db)

    if delta_database is not None:
        stamp = require_stamp(delta_database)
        spec = dispersion_utils.resolve(model_method, dispersion_kwargs, stamp=stamp)
        return {'database': delta_database, 'dispersion_kwargs': spec,
                'stamp': stamp, 'delta_source': delta_source}

    spec = dispersion_utils.resolve(model_method, dispersion_kwargs)
    database = prepare_delta_database(
        molecular_database,
        method=model_method,
        baseline_method=baseline_method,
        baseline_kwargs=baseline_kwargs,
        baseline_db=baseline_db,
        dispersion_db=dispersion_db,
        dispersion_kwargs=spec,
        property_to_learn=property_to_learn,
        xyz_derivative_property_to_learn=xyz_derivative_property_to_learn,
        delta_property=delta_property,
        delta_xyz_derivative_property=delta_xyz_derivative_property,
        qm_program=qm_program,
        working_directory=working_directory,
        verbose=verbose,
        subtract_atomic_shift=subtract_atomic_shift,
    )
    return {'database': database, 'dispersion_kwargs': spec,
            'stamp': stamp_of(database), 'delta_source': None}


def report_model_composition(baseline=None, neural_network='dNN',
                             dispersion_kwargs=None, verbose=1):
    """
    Print the energy expression of the fine-tuned model.

    Every defect in this area is invisible at the call site, so one line at the
    end of training makes the composition checkable at a glance.
    """
    expression = dispersion_utils.energy_expression(
        baseline=baseline, neural_network=neural_network,
        dispersion_kwargs=dispersion_kwargs)
    if verbose:
        print(f'\nFine-tuned model composition: {expression}')
    return expression
